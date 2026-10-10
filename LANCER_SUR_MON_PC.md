# Lancer l'optimisation du fichier 6 sur ton PC (16 cœurs + Gurobi)

## 1. Récupérer le code

```bash
git clone -b claude/elegant-cerf-3v2w6c https://github.com/yacineelborgi/EID.git
cd EID
```

(Ou sur GitHub : branche `claude/elegant-cerf-3v2w6c` → *Code* → *Download ZIP*.)

## 2. Installer les dépendances Python

```bash
python3 -m venv ~/eid-venv
source ~/eid-venv/bin/activate
pip install highspy scipy numpy ortools gurobipy sortedcontainers
```

## 3. Licence Gurobi (gratuite pour les étudiants)

1. Sur gurobi.com, connecte-toi avec ton compte étudiant → *Licenses* → *Request*.
2. Choisis **« WLS Academic »**. Elle marche partout tant que tu as internet.
3. Télécharge **`gurobi.lic`** et place-le dans ton dossier personnel : `~/gurobi.lic`.
4. Vérifie :
   ```bash
   python3 -c "import gurobipy as g; m=g.Model(); m.addVars(5000); m.update(); m.optimize(); print('Gurobi OK')"
   ```
   Si tu vois une erreur *size-limited license*, la licence n'est pas trouvée. Vérifie le chemin `~/gurobi.lic`.

(Variante : la licence *Named-User Academic* s'active avec `grbgetkey`. Il faut être sur le réseau de l'école ou sur son VPN.)

## 4. Lancer

```bash
chmod +x run_local.sh
./run_local.sh 4 4      # 4 recherches en parallèle × 4 threads = 16 cœurs
```

Le script choisit Gurobi tout seul si la licence fonctionne, sinon HiGHS.

- Suivre le meilleur score : `cat local/rr_shared.json.score`
- Voir l'avancement : `tail -f local/rr_1.log` (les lignes `NEW BEST` sont les améliorations)
- Arrêter : `pkill -f lns_rr.py`

## 5. Soumettre une amélioration

Le meilleur fichier est `local/rr_shared.json`. Vérifie-le avant de le soumettre :

```bash
python3 test_solution.py -d datasets/6_shortage.json -s local/rr_shared.json
```

Puis soumets-le sur Isograd comme fichier 6, et envoie-le-moi dans le chat pour que je reparte de lui.
