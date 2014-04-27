#include <vector>
#include <cstdio>
#include <CL/cl.h>

void readGraph(const char* filepath, std::vector<std::vector<cl_uint> >& edges) {
	FILE* fin;
	fin = fopen(filepath, "r");
	cl_uint x, y;
	cl_uint current_max_vertex = 0;
	while(EOF != fscanf(fin, "%d %d", &x, &y)) {
		if(x >= current_max_vertex) {
			current_max_vertex = x;
			edges.resize(x + 1);
		}
		if (y >= current_max_vertex) {
			current_max_vertex = y;
			edges.resize(y + 1);
		}
		edges[x].push_back(y);
		edges[y].push_back(x);
	}
	fclose(fin);
}

