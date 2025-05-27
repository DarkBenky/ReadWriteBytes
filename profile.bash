timeout 10s perf record -F 99 -g ./main
perf script > out.perf
~/Flamegraph/stackcollapse-perf.pl out.perf > out.folded
~/Flamegraph/flamegraph.pl out.folded > flamegraph.svg

echo "Flamegraph generated: flamegraph.svg"
xdg-open flamegraph.svg
