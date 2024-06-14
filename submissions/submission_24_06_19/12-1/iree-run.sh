../iree-build/tools/iree-run-module \
    --module=add_tosa.vmfb \
    --input="3x2xf32=[0 1][2 3][4 5]" \
    --input="1x1xf32=[10]"