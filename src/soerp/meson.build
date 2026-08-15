add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

f2py = find_program('f2py', required: true)

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

fortranobject_c = run_command(py,
  ['-c',
   'import numpy.f2py, os; print(os.path.join(numpy.f2py.get_include(), "fortranobject.c"))'],
  check: true
).stdout().strip()

py_mod_name = '_soerp'

fortran_sources = ['_soerp.f']
pyf_sources = ['_soerp.pyf']

f2py_target = custom_target(
  py_mod_name + '_f2py',
  input: pyf_sources,
  output: [py_mod_name + 'module.c', py_mod_name + '-f2pywrappers.f'],
  command: [f2py, '@INPUT@', '--lower', '--build-dir', '@OUTDIR@'],
)

# gfortran's runtime is not present on a stock Windows box, so link it in
# statically; otherwise the wheel imports only where MinGW is installed.
# Each flag is probed rather than assumed: -static-libquadmath only exists
# from GCC 9 onwards, and an unsupported flag is a hard link error.
fortran_link_args = []

if host_machine.system() == 'windows'
  cc = meson.get_compiler('c')
  foreach arg : ['-static-libgfortran', '-static-libgcc', '-static-libquadmath']
    if cc.has_link_argument(arg)
      fortran_link_args += arg
    endif
  endforeach
  fortran_link_args += ['-Wl,-Bstatic', '-lwinpthread', '-Wl,-Bdynamic']
endif

py.extension_module(
  py_mod_name,
  fortran_sources + [f2py_target, fortranobject_c],
  c_args: ['-I' + incdir_numpy, '-I' + incdir_f2py],
  link_args: fortran_link_args,
  dependencies: py_dep,
  install: true,
  subdir: 'soerp',
)

python_sources = [
  '__init__.py',
  'distributions.py',
  'method_of_moments.py',
  'statistics.py',
  'umath.py',
  'uncertain_function.py',
  'uncertain_variable.py',
]

py.install_sources(python_sources, subdir: 'soerp')
