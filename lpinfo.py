"""Write LP reduced profits per model: python lpinfo.py datasets/X.json margin out.json"""
import json
import sys

import mip4

inst = json.load(open(sys.argv[1]))
lpv, rc, ok, x = mip4.core_solve(inst, float(sys.argv[2]), 10)
json.dump({"lp": lpv, "ok": ok, "rc": rc}, open(sys.argv[3], "w"))
