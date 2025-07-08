pub use tch;

#[derive(Debug, Clone, Default)]
pub struct SpeechTimestamp {
    pub start: i64,
    pub end: i64,
}

#[derive(Debug, Clone)]
pub struct VadParams {
    pub threshold: f32,
    pub sampling_rate: usize,
    pub min_speech_duration_ms: usize,
    pub max_speech_duration_s: f32,
    pub min_silence_duration_ms: usize,
    pub speech_pad_ms: usize,
    pub return_seconds: bool,
    pub time_resolution: usize,
    pub visualize_probs: bool,
    pub neg_threshold: Option<f32>,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            sampling_rate: 16000,
            min_speech_duration_ms: 250,
            max_speech_duration_s: f32::INFINITY,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            return_seconds: false,
            time_resolution: 1,
            visualize_probs: false,
            neg_threshold: None,
        }
    }
}

pub type ProgressCallback = Box<dyn Fn(f32) + Send + Sync>;

pub trait VadModel {
    fn reset_states(&mut self);
    fn predict(&mut self, chunk: &[f32], sampling_rate: usize) -> Result<f32, tch::TchError>;
}

pub struct VadModelJit {
    model: tch::CModule,
    device: tch::Device,
}

impl VadModelJit {
    pub fn init_jit_model(model_path: &str, device: tch::Device) -> Result<Self, tch::TchError> {
        let mut model = tch::CModule::load_on_device(model_path, device).unwrap();
        model.set_eval();
        Ok(VadModelJit { model, device })
    }
}

impl VadModel for VadModelJit {
    fn reset_states(&mut self) {
        let _ = self
            .model
            .method_is::<tch::IValue>("reset_states", &[])
            .unwrap();
    }

    fn predict(&mut self, chunk: &[f32], sampling_rate: usize) -> Result<f32, tch::TchError> {
        let input_tensor = tch::Tensor::from_slice(chunk).to_device(self.device);
        let output = self.model.forward_is(&[
            tch::IValue::Tensor(input_tensor),
            tch::IValue::Int(sampling_rate as i64),
        ])?;
        if let tch::IValue::Tensor(output) = output {
            let prob = output.double_value(&[0]);
            Ok(prob as f32)
        } else {
            unreachable!("Expected output to be a tensor")
        }
    }
}

pub struct SileroVad<VAD: VadModel> {
    model: VAD,
}

#[derive(Debug, thiserror::Error)]
pub enum SileroVadErr {
    #[error("Failed on: {0}")]
    TchError(#[from] tch::TchError),
    #[error("Invalid sampling rate: {0}")]
    InvalidSamplingRate(String),
}

impl<VAD: VadModel> From<VAD> for SileroVad<VAD> {
    fn from(model: VAD) -> Self {
        SileroVad { model }
    }
}

impl<VAD: VadModel> SileroVad<VAD> {
    pub fn new(model: VAD) -> Self {
        Self { model }
    }

    pub fn get_speech_timestamps(
        &mut self,
        audio: Vec<f32>,
        params: VadParams,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<Vec<SpeechTimestamp>, SileroVadErr> {
        // ===== input validation and preprocessing =====

        let audio = audio;
        let sampling_rate = params.sampling_rate;

        if sampling_rate != 8000 && sampling_rate != 16000 {
            return Err(SileroVadErr::InvalidSamplingRate(
                "Sampling rate must be 8000 or 16000 Hz.".to_string(),
            ));
        }

        let window_size_samples = if sampling_rate == 16000 { 512 } else { 256 };

        self.model.reset_states();

        let min_speech_samples =
            (sampling_rate as f32 * params.min_speech_duration_ms as f32 / 1000.0) as usize;
        let speech_pad_samples =
            (sampling_rate as f32 * params.speech_pad_ms as f32 / 1000.0) as usize;
        let max_speech_samples = (sampling_rate as f32 * params.max_speech_duration_s
            - window_size_samples as f32
            - 2.0 * speech_pad_samples as f32) as usize;
        let min_silence_samples =
            (sampling_rate as f32 * params.min_silence_duration_ms as f32 / 1000.0) as usize;
        let min_silence_samples_at_max_speech = (sampling_rate as f32 * 98.0 / 1000.0) as usize; // 98ms

        let audio_length_samples = audio.len();

        // ===== step 1 =====

        let mut speech_probs = Vec::new();
        let mut current_start_sample = 0;

        while current_start_sample < audio_length_samples {
            let chunk_end = std::cmp::min(
                current_start_sample + window_size_samples,
                audio_length_samples,
            );

            let mut chunk = audio[current_start_sample..chunk_end].to_vec();

            // if chunk is shorter than window size, pad with zeros
            if chunk.len() < window_size_samples {
                chunk.resize(window_size_samples, 0.0);
            }

            let speech_prob = self.model.predict(&chunk, sampling_rate)?;
            speech_probs.push(speech_prob);

            // callback for progress
            let progress = current_start_sample + window_size_samples;
            let progress = std::cmp::min(progress, audio_length_samples);
            let progress_percent = (progress as f32 / audio_length_samples as f32) * 100.0;
            if let Some(ref callback) = progress_callback {
                callback(progress_percent);
            }

            current_start_sample += window_size_samples;
        }

        // ===== step 2 =====

        let mut triggered = false;
        let mut speeches = Vec::new();
        let mut current_speech = SpeechTimestamp::default();

        let neg_threshold = params
            .neg_threshold
            .unwrap_or_else(|| (params.threshold - 0.15).max(0.01));

        let mut temp_end = 0;
        let mut prev_end = 0;
        let mut next_start = 0;

        for (i, speech_prob) in speech_probs.iter().enumerate() {
            let current_sample = window_size_samples * i;

            // case 1: voice detected and previous temp_end exists
            if *speech_prob >= params.threshold && temp_end != 0 {
                temp_end = 0;
                if next_start < prev_end {
                    next_start = current_sample;
                }
            }

            // case 2: voice detected and not triggered
            if *speech_prob >= params.threshold && !triggered {
                triggered = true;
                current_speech.start = current_sample as i64;
                continue;
            }

            // case 3: voice segment is too long
            if triggered && (current_sample - current_speech.start as usize) > max_speech_samples {
                if prev_end > 0 {
                    // split at the previous end
                    current_speech.end = prev_end as i64;
                    speeches.push(current_speech.clone());
                    current_speech = SpeechTimestamp::default();
                    if next_start < prev_end {
                        triggered = false;
                    } else {
                        current_speech.start = next_start as i64;
                    }
                    prev_end = 0;
                    next_start = 0;
                    temp_end = 0;
                } else {
                    current_speech.end = current_sample as i64;
                    speeches.push(current_speech.clone());
                    current_speech = SpeechTimestamp::default();
                    prev_end = 0;
                    next_start = 0;
                    temp_end = 0;
                    triggered = false;
                }
                continue;
            }

            // case 4: silence detected
            if *speech_prob < neg_threshold && triggered {
                if temp_end == 0 {
                    temp_end = current_sample;
                }

                if (current_sample - temp_end) > min_silence_samples_at_max_speech {
                    prev_end = temp_end;
                }

                if (current_sample - temp_end) < min_silence_samples {
                    continue;
                } else {
                    current_speech.end = temp_end as i64;
                    if (current_speech.end - current_speech.start) > min_speech_samples as i64 {
                        speeches.push(current_speech.clone());
                    }
                    current_speech = SpeechTimestamp::default();
                    prev_end = 0;
                    next_start = 0;
                    temp_end = 0;
                    triggered = false;
                }
            }
        }

        // ===== handle the case where the last segment is still active =====

        if current_speech.start > 0
            && (audio_length_samples - current_speech.start as usize) > min_speech_samples
        {
            current_speech.end = audio_length_samples as i64;
            speeches.push(current_speech);
        }

        // ===== step 3 =====

        for i in 0..speeches.len() {
            if i == 0 {
                // first segment: add padding only at the start
                speeches[i].start = (speeches[i].start - speech_pad_samples as i64).max(0);
            }

            if i != speeches.len() - 1 {
                // not the last segment: handle the gap with the next segment
                let silence_duration = speeches[i + 1].start - speeches[i].end;
                if silence_duration < (2 * speech_pad_samples) as i64 {
                    speeches[i].end += silence_duration / 2;
                    speeches[i + 1].start = (speeches[i + 1].start - silence_duration / 2).max(0);
                } else {
                    speeches[i].end = (speeches[i].end + speech_pad_samples as i64)
                        .min(audio_length_samples as i64);
                    speeches[i + 1].start =
                        (speeches[i + 1].start - speech_pad_samples as i64).max(0);
                }
            } else {
                // last segment: add padding only at the end
                speeches[i].end =
                    (speeches[i].end + speech_pad_samples as i64).min(audio_length_samples as i64);
            }
        }

        // ===== step 4 =====

        if params.return_seconds {
            // convert start and end times to seconds
            let audio_length_seconds = audio_length_samples as f32 / sampling_rate as f32;
            for speech in &mut speeches {
                speech.start = ((speech.start as f32 / sampling_rate as f32).round() as i64).max(0);
                speech.end = ((speech.end as f32 / sampling_rate as f32).round() as i64)
                    .min(audio_length_seconds as i64);
            }
        }

        // ===== Optional: visualize probabilities =====

        if params.visualize_probs {
            self.visualize_probabilities(
                &speech_probs,
                window_size_samples as f32 / sampling_rate as f32,
            );
        }

        Ok(speeches)
    }

    fn visualize_probabilities(&self, probs: &[f32], time_step: f32) {
        println!("visualizing probabilities:");
        println!("times (s) | prob  | bar");
        println!("--------|------|--------");

        for (i, prob) in probs.iter().enumerate() {
            let time = i as f32 * time_step;
            let bar_length = (*prob * 50.0) as usize;
            let bar = "█".repeat(bar_length);
            println!("{:7.2} | {:.3} | {}", time, prob, bar);
        }
    }
}

// ===== usage example =====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_basic() {
        let model_path = std::env::var("VAD_MODEL_PATH").unwrap();
        let model = VadModelJit::init_jit_model(&model_path, tch::Device::Cpu).unwrap();

        let mut vad = SileroVad::new(model);

        // 创建测试音频（1秒的随机音频）
        let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.001).sin()).collect();

        let params = VadParams {
            threshold: 0.5,
            sampling_rate: 16000,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            return_seconds: true,
            ..Default::default()
        };

        let progress_callback = Some(Box::new(|progress: f32| {
            println!("handling progress: {:.1}%", progress);
        }) as ProgressCallback);

        match vad.get_speech_timestamps(audio, params, progress_callback) {
            Ok(speeches) => {
                println!("detected {} speech segments:", speeches.len());
                for (i, speech) in speeches.iter().enumerate() {
                    println!(
                        "segments {}: {:.3}s - {:.3}s",
                        i + 1,
                        speech.start,
                        speech.end
                    );
                }
            }
            Err(e) => {
                eprintln!("VAD Error: {}", e);
            }
        }
    }
}
