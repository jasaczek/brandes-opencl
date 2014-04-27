Zaimplementowany algorytm jest tym, który jest opisany w Betweenness
Centrality on GPUs and Heterogeneous Architectures, a dokładnie jego wersja z
wirtualizacją wierzchołków i Stride-CSR representation. Kompresja grafu nie
została zaimplementowana.

Zastosowane optymalizacje to są właśnie te opisane w paperze.
Rozmiaw work-groupy został ustalony na 128 gdyż eksperymentalnie na takim
rozmiarze otrzymałem najlepsze wyniki. W pliku test_results.txt są wypisane
czasy działań na różnych wielkościach grup. Aby możliwe byłe ustawienie
względnie dowolnej wielkości grupy sztucznie powiększam globalną liczbę
work-itemów z których część nic nie robi.

Ten katalog jest repozytorium git-a. I wszystkie zmiany są widoczne w
historii.
