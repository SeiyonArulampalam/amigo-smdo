@echo off
:: Fix up environment variables that conda-forge packages set from %PREFIX%,
:: which is only defined during a conda *build*. Under pixi they would expand
:: to nothing (e.g. MSMPI_INC=\Library\include), and CMake's FindMPI then
:: builds a broken MPI::MPI_CXX imported target.
::
:: Also point FindMETIS at the environment. These are needed both to build
:: amigo and at run time, because model.build_module() re-runs CMake, and
:: amigoConfig.cmake calls find_dependency(MPI) / find_package(METIS).
set "MSMPI_BIN=%CONDA_PREFIX%\Library\bin"
set "MSMPI_INC=%CONDA_PREFIX%\Library\include"
set "MSMPI_LIB64=%CONDA_PREFIX%\Library\lib"
set "METIS_ROOT=%CONDA_PREFIX%\Library"
