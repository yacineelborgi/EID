# Preuve d'optimalité de 4_precise (1 031 941)

Aucune solution légitime ne dépasse 1 031 941 sur `4_precise`.

1. Le modèle agrégé (énergie + données libres par type) est une relaxation : sans modèle
   mono-type, il n'y a pas de source possible.
2. Avec les duaux LP exacts y : valeur = U − Σ coûts de déviation − y·marges, où U = 1 031 947,98.
   Atteindre 942 impose donc un coût total ≤ 5,985 : seules 360 déviations sont candidates.
3. Recherche exhaustive meet-in-the-middle en C (`mm.c`) sur toutes les combinaisons
   (~4·10¹³ sous-ensembles de coût ≤ 5,985) : 0 solution ≥ 942.
   Test de contrôle à la cible 941 : retrouve exactement la solution soumise.

Deux autres implémentations indépendantes (découpages différents) donnent le même résultat.

Utilisation : `python3 setup.py`, `python3 prep.py <cible> <marge> <sortie>`,
`gcc -O3 -o mm mm.c`, `./mm <entrée> 1|2 <max_hits>`, `python3 decode.py`.
Les scripts écrivent et lisent leurs fichiers intermédiaires dans le répertoire courant :
lance-les depuis `proof_precise/`.
