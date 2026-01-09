project('dot', 'cpp', 'cuda',
  version : '0.1.0',
  default_options : [
    'cpp_std=c++17',
    'cuda_std=c++17',
    'buildtype=release',
    'warning_level=2',
  ]
)

# Find Python and PyTorch
# pure: false ensures all files are installed to platlib (with the .so)
python = import('python').find_installation('python3', pure: false)
python_dep = python.dependency()

# Get PyTorch paths from Python
torch_include_result = run_command(python, '-c',
  'import torch; from torch.utils.cpp_extension import include_paths; print(";".join(include_paths("cuda")))',
  check : true
)
torch_lib_result = run_command(python, '-c',
  'import torch; from torch.utils.cpp_extension import library_paths; print(";".join(library_paths("cuda")))',
  check : true
)
torch_abi_result = run_command(python, '-c',
  'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))',
  check : true
)

torch_includes = torch_include_result.stdout().strip().split(';')
torch_lib_dirs = torch_lib_result.stdout().strip().split(';')
torch_abi = torch_abi_result.stdout().strip()

message('PyTorch include paths: ' + torch_include_result.stdout().strip())
message('PyTorch library paths: ' + torch_lib_result.stdout().strip())
message('PyTorch CXX11 ABI: ' + torch_abi)

# Create include directories
torch_inc_dirs = []
foreach inc : torch_includes
  torch_inc_dirs += include_directories(inc, is_system : true)
endforeach

# Add local src directory for common/ headers
# Allows: #include "common/numerics.cuh"
src_inc = include_directories('src')

# Find torch libraries (CPU and CUDA)
cpp = meson.get_compiler('cpp')
torch_lib = cpp.find_library('torch', dirs : torch_lib_dirs, required : true)
torch_cpu_lib = cpp.find_library('torch_cpu', dirs : torch_lib_dirs, required : true)
torch_cuda_lib = cpp.find_library('torch_cuda', dirs : torch_lib_dirs, required : true)
c10_lib = cpp.find_library('c10', dirs : torch_lib_dirs, required : true)
c10_cuda_lib = cpp.find_library('c10_cuda', dirs : torch_lib_dirs, required : true)

torch_deps = [torch_lib, torch_cpu_lib, torch_cuda_lib, c10_lib, c10_cuda_lib]

# Compiler flags
cxx_abi_flag = '-D_GLIBCXX_USE_CXX11_ABI=' + torch_abi
cxx_args = ['-O3', cxx_abi_flag, '-DUSE_TORCH_LIBRARY']
cuda_args = ['-O3', '-DUSE_TORCH_LIBRARY', '--expt-relaxed-constexpr']

# Source files
cpp_sources = files(
  # Sinkhorn module
  'src/sinkhorn/registry.cpp',
  'src/sinkhorn/torch_cuda.cpp',
  'src/sinkhorn/torch_cpu.cpp',
  'src/sinkhorn/kernels_cpu.cpp',
)

cuda_sources = files(
  'src/sinkhorn/kernels.cu',
)

# Build Python extension module
python.extension_module('_C',
  cpp_sources + cuda_sources,
  include_directories : [src_inc] + torch_inc_dirs,
  dependencies : [python_dep] + torch_deps,
  cpp_args : cxx_args,
  cuda_args : cuda_args,
  install : true,
  subdir : 'dot',
)

# Install Python source files
python.install_sources(
  'dot/__init__.py',
  'dot/_ops.py',
  'dot/sinkhorn.py',
  subdir : 'dot',
)
