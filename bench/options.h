#pragma once
#include <iostream>

#include "argh.h"

struct options {
	bool show_image = false;
	bool kernel_perf = false;
	bool per_iter_perf = false;
	int nx = 500;
	int ny = 500;
	int ns = 1;
	int max_depth = 50;
};

bool parse_args(options& o, int argc, char** argv) {
	argh::parser cmdl(argv);
	if (cmdl["help"]) {
		std::cerr << "usage [-nx=<image-width 500>] [-ny=<image-height 500>] [-ns=<spp 1>] [-md=<max-depth 50>] [-show_image] [-kernel_perf] [-per_iter_perf]" << std::endl;
		return false;
	}
	cmdl({ "nx", "width" }, o.nx) >> o.nx;
	cmdl({ "ny", "height" }, o.ny) >> o.ny;
	cmdl({ "ns", "num_samples" }, o.ns) >> o.ns;
	cmdl({ "md", "max_depth" }, o.max_depth) >> o.max_depth;
	o.show_image = cmdl["show_image"];
	o.kernel_perf = cmdl["kernel_perf"];
	o.per_iter_perf = cmdl["per_iter_perf"];
}
