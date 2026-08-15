project(
    'soerp',
    'c', 'fortran',
    version: run_command(
        ['python', 'bin/get_version.py'], check: true
    ).stdout().strip(),
    meson_version: '>=1.1.0',
    default_options: ['buildtype=release', 'warning_level=1'],
)

subdir('src')
