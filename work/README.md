# Outils de travail des agents (sauvegarde)

Copie des scripts et plans produits pendant l'optimisation (scratchpad de session).
Les chemins absolus `/tmp/claude-0/.../scratchpad/<dossier>` dans ces scripts correspondent
à `work/<dossier>` ici : adapter avant de relancer.

- `s6/`, `shortA/`, `shortA2/`, `shortB2/` : 6_shortage (mip6 multi-seuils, certify, fix.py = meilleur pipeline 669 445).
- `big7/` : 7_big (pack7b.py, prove7b.py). `e5/` : 5_energy (pack5.py, relax.py).
- `med/`, `medEnergy/`, `sA/`, `stratB/`, `lsC/`, `D/` : 2_medium (plans, recherche ciblée, LNS).

Pistes ouvertes : voir le tableau des résultats dans le README racine. Le fichier 6 est celui où il
reste le plus à gagner (record 669 772, nous 669 445, plafond prouvé 669 934).
