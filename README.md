# Silero-VAD V5 in Rust (based on LibTorch)

This is a Rust implementation of [Silero-VAD V5](https://github.com/snakers4/silero-vad), **rewritten by Claude from the original Python version**. Silero-VAD is a Voice Activity Detection (VAD) system that can detect speech segments in audio files, separating speech from silence and noise.

The implementation uses LibTorch for PyTorch model inference and maintains full compatibility with the original Python version's results. It provides the same functionality including speech detection, timestamp generation, and configurable parameters for different use cases.

## What is VAD?

Voice Activity Detection (VAD) is used to:
- **Detect speech segments** in audio recordings
- **Generate timestamps** for when speech starts and ends
- **Remove silence** from audio for further processing
- **Preprocess audio** for speech recognition systems
- **Analyze conversations** and meetings

## Requirements

- **Rust**: 1.70.0 or later
- **LibTorch**: 1.13.0 or later
- **System**: GCC 11.4.0+ (Linux), MSVC 2019+ (Windows), or Xcode Command Line Tools (macOS)

## LibTorch Installation

### Download LibTorch

#### Linux
```bash
# CPU Version
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cpu.zip

# CUDA Version
wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu128.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu128.zip
```

#### Windows
```powershell
# CPU Version
Invoke-WebRequest -Uri "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.0%2Bcpu.zip" -OutFile "libtorch-cpu.zip"
Expand-Archive -Path "libtorch-cpu.zip" -DestinationPath "."

# CUDA Version
Invoke-WebRequest -Uri "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.4.0%2Bcu128.zip" -OutFile "libtorch-cuda.zip"
Expand-Archive -Path "libtorch-cuda.zip" -DestinationPath "."
```

#### macOS
```bash
# CPU Version
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.0.zip
unzip libtorch-macos-arm64-2.4.0.zip
```

### Set Environment Variables

#### Linux/macOS
```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=${LIBTORCH}/lib:$DYLD_LIBRARY_PATH  # macOS
```

#### Windows
```powershell
$env:LIBTORCH = "C:\path\to\libtorch"
$env:PATH += ";C:\path\to\libtorch\lib"
```

## Download Model

```bash
# Download the PyTorch JIT model
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.jit
```

## About

This Rust implementation was created by Claude AI as a faithful port of the original Python Silero-VAD implementation. It maintains the same algorithm logic, parameter handling, and output format while leveraging Rust's performance and safety features.

The implementation includes all features from the Python version:
- Speech timestamp detection
- Configurable thresholds and durations
- Speech padding
- Progress tracking
- Multiple output formats

## License

This project follows the same license as the original Silero-VAD project.