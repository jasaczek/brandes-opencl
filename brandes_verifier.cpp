#include <cstdio>
#include <vector>
#include <stack>
#include <queue>

using std::vector;
using std::stack;
using std::priority_queue;

vector<vector<int> > edges;
int current_max_vertex = -1;

int main(int argc, char* argv[]) {
	FILE* fin;
	fin = fopen(argv[1], "r");
	int x, y;
	while(EOF != fscanf(fin, "%d %d", &x, &y)) {
		if(x > current_max_vertex) {
			current_max_vertex = x;
			edges.resize(x + 1);
		}
		if (y > current_max_vertex) {
			current_max_vertex = y;
			edges.resize(y + 1);
		}
		edges[x].push_back(y);
		edges[y].push_back(x);
	}
	fclose(fin);

	vector<double> Cb(edges.size(), 0);
	for(int s = 0; s < edges.size(); ++s) {
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

		vector<double> delta(edges.size(), 0);
		while(!stack.empty()) {
			int w = stack.top();
			stack.pop();
			for(auto v = precedessors[w].begin(); v != precedessors[w].end(); ++v) {
				delta[*v] = delta[*v] + (((double) sigma[*v]) / ((double) sigma[w])) * (1.0 + delta[w]);
			}
			if (w != s) {
				Cb[w] = Cb[w] + delta[w];
			}
		}
	}

	for(int i = 0; i < edges.size(); ++i) {
		printf("%f\n", Cb[i]);
	}
	return 0;
}
