#include <cstdio>
#include <vector>
#include <stack>
#include <queue>
#include "brandes_utils.hpp"

using std::vector;
using std::stack;
using std::priority_queue;

vector<vector<cl_uint> > edges;

int main(int argc, char* argv[]) {
	readGraph(argv[1], edges);

	vector<float> Cb(edges.size(), 0);
	for(size_t s = 0; s < edges.size(); ++s) {
		stack<int> stack;
		vector<vector<int> > precedessors(edges.size());
		vector<int> sigma(edges.size(), 0);
		sigma[s] = 1;
		vector<int> dist(edges.size(), -1);
		dist[s] = 0;

		priority_queue<std::pair<int, int> > queue;
		queue.emplace(0, s);
		while(!queue.empty()) {
			int v = queue.top().second;
			queue.pop();
			stack.push(v);
			for(auto w = edges[v].begin(); w != edges[v].end(); ++w) {
				if(dist[*w] < 0) {
					dist[*w] = dist[v] + 1;
					queue.emplace(-dist[*w], *w);
				}
				if (dist[*w] == dist[v] + 1) {
					sigma[*w] = sigma[*w] + sigma[v];
					precedessors[*w].push_back(v);
				}
			}
		}

		vector<float> delta(edges.size(), 0);
		while(!stack.empty()) {
			size_t w = stack.top();
			stack.pop();
			for(auto v = precedessors[w].begin(); v != precedessors[w].end(); ++v) {
				delta[*v] = delta[*v] + (((float) sigma[*v]) / ((float) sigma[w])) * (1.0 + delta[w]);
			}
			if (w != s) {
				Cb[w] = Cb[w] + delta[w];
			}
		}
	}

	for(size_t i = 0; i < edges.size(); ++i) {
		printf("%f\n", Cb[i]);
	}
	return 0;
}
