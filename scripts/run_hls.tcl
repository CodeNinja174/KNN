open_project knn_hls_accelerator
set_top knn_hls_top
add_files hls/knn_hls_top.cpp
add_files hls/knn_hls.hpp
open_solution "solution1" -flow_target vivado
set_part xczu28dr-ffvg1517-2-e
create_clock -period 5 -name default
csynth_design
export_design -format ip_catalog
exit
