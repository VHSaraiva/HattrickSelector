# nt_selector_formations_with_foxtrick.py
# Best XI (HO!/Java engine) + Foxtrick-style "Top 10 contributors" per role
# Upload NT_Prospects.csv (no Position needed). Everyone is rated for each role.

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Player model & CSV loader
# =========================

@dataclass
class Player:
    Name: str = ""
    Behaviour: str = "Normal"
    Side: str = ""
    Specialty: str = "None"
    KeeperSkill: float = 0.0
    DefenderSkill: float = 0.0
    PlaymakerSkill: float = 0.0
    WingerSkill: float = 0.0
    PassingSkill: float = 0.0
    ScorerSkill: float = 0.0
    Form: float = 7.0
    Experience: float | str = 0.0
    Stamina: float = 7.0
    Loyalty: float = 0.0
    HomeGrown: int = 0
    Age: float | None = None  # optional, used only for filters

CSV_COL_MAP = {
    "Name":"Name","Player":"Name",
    "Behaviour":"Behaviour","Behavior":"Behaviour",
    "Side":"Side",
    "Speciality":"Specialty","Specialty":"Specialty",
    "KeeperSkill":"KeeperSkill","Goalkeeping":"KeeperSkill",
    "DefenderSkill":"DefenderSkill","Defending":"DefenderSkill",
    "PlaymakerSkill":"PlaymakerSkill","Playmaking":"PlaymakerSkill",
    "WingerSkill":"WingerSkill","Winger":"WingerSkill",
    "PassingSkill":"PassingSkill","Passing":"PassingSkill",
    "ScorerSkill":"ScorerSkill","Scoring":"ScorerSkill",
    "Form":"Form","PlayerForm":"Form",
    "Stamina":"Stamina","StaminaSkill":"Stamina",
    "Experience":"Experience",
    "Loyalty":"Loyalty",
    "HomeGrown":"HomeGrown","Homegrown":"HomeGrown",
    "Age":"Age","PlayerAge":"Age",
}

def _f(x, d=0.0):
    try: return float(x)
    except: 
        try:
            s = str(x).strip()
            return float(s) if s else d
        except:
            return d

def load_players_from_csv(file) -> List[Player]:
    df = pd.read_csv(file)
    df = df.rename(columns={c: CSV_COL_MAP.get(c, c) for c in df.columns})
    out: List[Player] = []
    for _, r in df.iterrows():
        age_val = r.get("Age", None)
        try:
            age = float(age_val) if age_val is not None and str(age_val).strip() != "" else None
        except:
            age = None
        out.append(Player(
            Name=str(r.get("Name","")),
            Behaviour=str(r.get("Behaviour","Normal")),
            Side=str(r.get("Side","")),
            Specialty=str(r.get("Specialty","None")),
            KeeperSkill=_f(r.get("KeeperSkill",0)),
            DefenderSkill=_f(r.get("DefenderSkill",0)),
            PlaymakerSkill=_f(r.get("PlaymakerSkill",0)),
            WingerSkill=_f(r.get("WingerSkill",0)),
            PassingSkill=_f(r.get("PassingSkill",0)),
            ScorerSkill=_f(r.get("ScorerSkill",0)),
            Form=_f(r.get("Form",7)),
            Experience=r.get("Experience",0),
            Stamina=_f(r.get("Stamina",7)),
            Loyalty=_f(r.get("Loyalty",0)),
            HomeGrown=int(r.get("HomeGrown",0) or 0),
            Age=age,
        ))
    return out

# =========================
# Core maths (HO!/Java model) – for Best XI
# =========================

NORMAL, OFFENSIVE, DEFENSIVE, TOWARDS_MIDDLE, TOWARDS_WING = \
    "NORMAL","OFFENSIVE","DEFENSIVE","TOWARDS_MIDDLE","TOWARDS_WING"
BEH_MAP = {
    "Normal": NORMAL, "Offensive": OFFENSIVE, "Defensive": DEFENSIVE,
    "TW": TOWARDS_WING, "Towards Wing": TOWARDS_WING,
    "TM": TOWARDS_MIDDLE, "Towards Middle": TOWARDS_MIDDLE,
}

def skill_rating(x: float) -> float: return max(0.0, float(x) - 1.0)
def loyalty_term(hg: int, loy: float) -> float: return 1.5 if int(hg) else skill_rating(loy)/19.0
def form_factor(f: float) -> float:
    f = min(7.0, skill_rating(f))
    return 0.378 * math.sqrt(max(0.0, f))
def strength_generic(sv: float, hg: int, loy: float, form: float) -> float:
    return (skill_rating(sv) + loyalty_term(hg, loy)) * form_factor(form)

def weather_factor(spec: str, weather: str) -> float:
    s = (spec or "None").strip()
    if s == "Technical": return 0.95 if weather == "RAINY" else 1.05 if weather == "SUNNY" else 1.0
    if s == "Powerful":  return 1.05 if weather == "RAINY" else 0.95 if weather == "SUNNY" else 1.0
    if s == "Quick":     return 0.95 if weather in ("RAINY","SUNNY") else 1.0
    return 1.0

def stamina_factor(stamina: float, minute: int, start: int, tactic: str) -> float:
    press = 1.1 if tactic == "Pressing" else 1.0
    s = skill_rating(float(stamina))
    if s < 7: r0 = 102 + (23/7)*s; delta = press * ((27/70)*s - 5.95)
    else:     r0 = 125 + (s-7)*(100/7); delta = -3.25*press
    r = r0
    to = min(45, minute)
    if start < to: r += (to-start)*delta/5
    if minute >= 45:
        if start < 45: r = min(r0, r+18.75)
        frm = max(45,start); to = min(90,minute)
        if frm < to: r += (to-frm)*delta/5
    if minute >= 90:
        if start < 90: r = min(r0, r+6.25)
        frm = max(90,start)
        if frm < minute: r += (minute-frm)*delta/5
    return min(1.0, r/100.0)

def trainer_factor_def(coach: int) -> float:
    return 1.02 - coach*(1.15-1.02)/10 if coach <= 0 else 1.02 - coach*(1.02-0.90)/10
def trainer_factor_att(coach: int) -> float:
    return 1.02 - coach*(0.90-1.02)/10 if coach <= 0 else 1.02 - coach*(1.02-1.10)/10
def mf_context(attitude: str, location: str, tactic: str, ts: float) -> float:
    r = 1.0
    if attitude == "PIC": r *= 0.83945
    elif attitude == "MOTS": r *= 1.1149
    if location == "HOME": r *= 1.19892
    elif location == "AWAY_DERBY": r *= 1.11493
    if tactic == "CounterAttacks": r *= 0.93
    elif tactic == "LongShots": r *= 0.96
    r *= 0.1 + 0.425 * math.sqrt(max(0.0, float(ts)))
    return r
def def_context(sector: str, tactic: str, coach: int) -> float:
    r = trainer_factor_def(coach)
    if sector in ("DL","DR"):
        if tactic == "AttackInTheMiddle": r *= 0.85
        elif tactic == "PlayCreatively": r *= 0.93
    else:
        if tactic == "AttackInWings": r *= 0.85
        elif tactic == "PlayCreatively": r *= 0.93
    return r
def att_context(tactic: str, coach: int, confidence: float) -> float:
    r = trainer_factor_att(coach)
    if tactic == "LongShots": r *= 0.96
    r *= 0.8 + 0.05 * (float(confidence)+0.5)
    return r

SCALE = {"MF":0.312,"DL":0.834,"DR":0.834,"DC":0.501,"AC":0.513,"AL":0.615,"AR":0.615}
def to_ht_bar(sector: str, raw_sum: float) -> float:
    if raw_sum <= 0: return 0.75
    x = raw_sum * SCALE[sector]
    return (x**1.2)/4.0 + 1.0

def strength_for(p: Player, skill: str) -> float:
    return strength_generic({
        "KEEPER": p.KeeperSkill,
        "DEFENDING": p.DefenderSkill,
        "PLAYMAKING": p.PlaymakerSkill,
        "WINGER": p.WingerSkill,
        "PASSING": p.PassingSkill,
        "SCORING": p.ScorerSkill,
    }[skill], p.HomeGrown, p.Loyalty, p.Form)

def beh_weights(**k):
    base = {NORMAL:0.0,OFFENSIVE:0.0,DEFENSIVE:0.0,TOWARDS_MIDDLE:0.0,TOWARDS_WING:0.0}
    base.update(k); return base

def java_entries():
    E=[]; add=lambda g,s,sec,side,beh,spec=None:E.append((g,s,sec,side,beh,spec))
    # SD
    add("SD","KEEPER","Goal","NONE",beh_weights(**{NORMAL:.61}))
    add("SD","DEFENDING","Goal","NONE",beh_weights(**{NORMAL:.25}))
    add("SD","DEFENDING","CentralDefence","THIS_SIDE_ONLY",beh_weights(**{NORMAL:.52,OFFENSIVE:.4,TOWARDS_WING:.81}))
    add("SD","DEFENDING","CentralDefence","MIDDLE_ONLY",beh_weights(**{NORMAL:.26,OFFENSIVE:.20}))
    add("SD","DEFENDING","Back","THIS_SIDE_ONLY",beh_weights(**{NORMAL:.92,OFFENSIVE:.74,DEFENSIVE:1.0,TOWARDS_MIDDLE:.75}))
    add("SD","DEFENDING","InnerMidfield","THIS_SIDE_ONLY",beh_weights(**{NORMAL:.19,OFFENSIVE:.09,DEFENSIVE:.27,TOWARDS_WING:.24}))
    add("SD","DEFENDING","InnerMidfield","MIDDLE_ONLY",beh_weights(**{NORMAL:.095,OFFENSIVE:.045,DEFENSIVE:.135}))
    add("SD","DEFENDING","Wing","THIS_SIDE_ONLY",beh_weights(**{NORMAL:.35,OFFENSIVE:.22,DEFENSIVE:.61,TOWARDS_MIDDLE:.29}))
    # CD
    add("CD","KEEPER","Goal","NONE",beh_weights(**{NORMAL:.87}))
    add("CD","DEFENDING","Goal","NONE",beh_weights(**{NORMAL:.35}))
    add("CD","DEFENDING","CentralDefence","NONE",beh_weights(**{NORMAL:1.0,OFFENSIVE:.73,TOWARDS_WING:.67}))
    add("CD","DEFENDING","Back","NONE",beh_weights(**{NORMAL:.38,OFFENSIVE:.35,DEFENSIVE:.43,TOWARDS_MIDDLE:.7}))
    add("CD","DEFENDING","InnerMidfield","NONE",beh_weights(**{NORMAL:.4,OFFENSIVE:.16,DEFENSIVE:.58,TOWARDS_WING:.33}))
    add("CD","DEFENDING","Wing","NONE",beh_weights(**{NORMAL:.2,OFFENSIVE:.13,DEFENSIVE:.25,TOWARDS_MIDDLE:.25}))
    # MF
    add("MF","PLAYMAKING","CentralDefence","NONE",beh_weights(**{NORMAL:.25,OFFENSIVE:.4,TOWARDS_WING:.15}))
    add("MF","PLAYMAKING","Back","NONE",beh_weights(**{NORMAL:.15,OFFENSIVE:.2,DEFENSIVE:.1,TOWARDS_MIDDLE:.2}))
    add("MF","PLAYMAKING","InnerMidfield","NONE",beh_weights(**{NORMAL:1.0,OFFENSIVE:.95,DEFENSIVE:.95,TOWARDS_WING:.9}))
    add("MF","PLAYMAKING","Wing","NONE",beh_weights(**{NORMAL:.45,OFFENSIVE:.3,DEFENSIVE:.3,TOWARDS_MIDDLE:.55}))
    add("MF","PLAYMAKING","Forward","NONE",beh_weights(**{NORMAL:.25,DEFENSIVE:.35,TOWARDS_WING:.15}))
    # CA
    add("CA","PASSING","InnerMidfield","NONE",beh_weights(**{NORMAL:.33,OFFENSIVE:.49,DEFENSIVE:.18,TOWARDS_WING:.23}))
    add("CA","PASSING","Wing","NONE",beh_weights(**{NORMAL:.11,OFFENSIVE:.13,DEFENSIVE:.05,TOWARDS_MIDDLE:.16}))
    add("CA","PASSING","Forward","NONE",beh_weights(**{NORMAL:.33,DEFENSIVE:.53,TOWARDS_WING:.23}))
    add("CA","SCORING","InnerMidfield","NONE",beh_weights(**{NORMAL:.22,OFFENSIVE:.31,DEFENSIVE:.13}))
    add("CA","SCORING","Forward","NONE",beh_weights(**{NORMAL:1.0,DEFENSIVE:.56,TOWARDS_WING:.66}))
    # SA
    add("SA","PASSING","InnerMidfield","MIDDLE_ONLY",beh_weights(**{NORMAL:.13,OFFENSIVE:.18,DEFENSIVE:.07}))
    add("SA","PASSING","Forward","NONE",beh_weights(**{NORMAL:.14,DEFENSIVE:.31}))
    add("SA","PASSING","Forward","NONE",{NORMAL:0,OFFENSIVE:0,DEFENSIVE:.41,TOWARDS_MIDDLE:0,TOWARDS_WING:0},"Technical")
    add("SA","PASSING","InnerMidfield","THIS_SIDE_ONLY",beh_weights(**{NORMAL:.26,OFFENSIVE:.36,DEFENSIVE:.14,TOWARDS_WING:.31}))
    add("SA","PASSING","Wing","THIS_SIDE_ONLY",beh_weights(**{NORMAL:.26,OFFENSIVE:.29,DEFENSIVE:.21,TOWARDS_MIDDLE:.15}))
    add("SA","PASSING","Forward","THIS_SIDE_ONLY",beh_weights(**{TOWARDS_WING:.21}))
    add("SA","PASSING","Forward","OPPOSITE_SIDE_ONLY",beh_weights(**{TOWARDS_WING:.06}))
    add("SA","WINGER","CentralDefence","THIS_SIDE_ONLY",beh_weights(**{TOWARDS_WING:.26}))
    add("SA","WINGER","Back","THIS_SIDE_ONLY",beh_weights(**{NORMAL:.59,OFFENSIVE:.69,DEFENSIVE:.45,TOWARDS_MIDDLE:.35}))
    add("SA","WINGER","InnerMidfield","THIS_SIDE_ONLY",beh_weights(**{TOWARDS_WING:.59}))
    add("SA","WINGER","Wing","THIS_SIDE_ONLY",beh_weights(**{NORMAL:.86,OFFENSIVE:1.0,DEFENSIVE:.69,TOWARDS_MIDDLE:.74}))
    add("SA","WINGER","Forward","NONE",beh_weights(**{NORMAL:.24,DEFENSIVE:.13}))
    add("SA","WINGER","Forward","THIS_SIDE_ONLY",beh_weights(**{TOWARDS_WING:.64}))
    add("SA","WINGER","Forward","OPPOSITE_SIDE_ONLY",beh_weights(**{TOWARDS_WING:.21}))
    add("SA","SCORING","Forward","NONE",beh_weights(**{NORMAL:.27,DEFENSIVE:.13}))
    add("SA","SCORING","Forward","OPPOSITE_SIDE_ONLY",beh_weights(**{TOWARDS_WING:.19}))
    add("SA","SCORING","Forward","THIS_SIDE_ONLY",beh_weights(**{TOWARDS_WING:.51}))
    return E

GROUP_TO_SECTORS = {"SD":("DL","DR"),"CD":("DC",),"MF":("MF",),"SA":("AL","AR"),"CA":("AC",)}
ROLE_DEF = {
    "CD": ("CD","M","DC"),
    "RB": ("WB","R","DR"),
    "LB": ("WB","L","DL"),
    "IM": ("IM","M","MF"),
    "RW": ("WI","R","AR"),
    "LW": ("WI","L","AL"),
    "FW": ("FW","M","AC"),
}

def side_ok(restr: str, target_side: str, player_side: str) -> bool:
    if restr == "NONE": return True
    if restr == "MIDDLE_ONLY": return player_side == "M"
    if restr == "THIS_SIDE_ONLY": return player_side == target_side
    if restr == "OPPOSITE_SIDE_ONLY":
        return (target_side == "L" and player_side == "R") or (target_side == "R" and player_side == "L")
    return True

def exp_term(exp_val: float | str, sector_key: str) -> float:
    try: e = float(exp_val)
    except: e = float(str(exp_val).strip() or 0)
    e = skill_rating(e)
    k = (-0.00000725*(e**4) + 0.0005*(e**3) - 0.01336*(e**2) + 0.176*e)
    mult = {"DL":0.345,"DR":0.345,"DC":0.48,"MF":0.73,"AL":0.375,"AR":0.375,"AC":0.45}
    return k*mult.get(sector_key,0.0)

def lineup_sector_by_role(test_role: str) -> str:
    return {"GK":"Goal","CD":"CentralDefence","WB":"Back","IM":"InnerMidfield","WI":"Wing","FW":"Forward"}[test_role]

def player_sector_bar_single(p: Player, test_role: str, test_side: str, target_key: str,
                             minute: int, tactic: str, weather: str,
                             attitude: str, location: str, coach: int, ts: float, conf: float) -> float:
    entries = java_entries()
    tgt_side = "M"
    if target_key in ("DL","AL"): tgt_side = "L"
    if target_key in ("DR","AR"): tgt_side = "R"
    beh = BEH_MAP.get(p.Behaviour, NORMAL)
    pl_sector = lineup_sector_by_role(test_role)
    psum = 0.0
    for (group, skill, sector, side_restr, beh_w, spec_only) in entries:
        if target_key not in GROUP_TO_SECTORS[group]: continue
        if sector != pl_sector: continue
        if not side_ok(side_restr, tgt_side, test_side): continue
        if spec_only and p.Specialty != spec_only: continue
        w = beh_w.get(beh, beh_w.get(NORMAL, 0.0))
        if w == 0.0: continue
        psum += w * strength_for(p, skill)
    if psum <= 0.0: return 0.75
    psum += exp_term(p.Experience, target_key)
    psum *= weather_factor(p.Specialty, weather)
    psum *= stamina_factor(p.Stamina, minute, 0, tactic)
    if target_key == "MF": psum *= mf_context(attitude, location, tactic, ts)
    elif target_key in ("DL","DR","DC"): psum *= def_context(target_key, tactic, coach)
    else: psum *= att_context(tactic, coach, conf)
    return to_ht_bar(target_key, psum)

# =========================
# Hungarian (maximize) – for Best XI assignment
# =========================

def hungarian_maximize(score_matrix: np.ndarray):
    maxv = float(np.max(score_matrix)) if score_matrix.size else 0.0
    cost = maxv - score_matrix
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    pad = np.zeros((n,n)); pad[:n_rows,:n_cols] = cost

    def hungarian(a: np.ndarray):
        a = a.copy(); n = a.shape[0]
        u = np.zeros(n+1); v = np.zeros(n+1); p = np.zeros(n+1, dtype=int); way = np.zeros(n+1, dtype=int)
        for i in range(1, n+1):
            p[0] = i; j0 = 0
            minv = np.full(n+1, np.inf); used = np.zeros(n+1, dtype=bool)
            while True:
                used[j0] = True; i0 = p[j0]; delta = np.inf; j1 = 0
                for j in range(1, n+1):
                    if used[j]: continue
                    cur = a[i0-1, j-1] - u[i0] - v[j]
                    if cur < minv[j]: minv[j] = cur; way[j] = j0
                    if minv[j] < delta: delta = minv[j]; j1 = j
                for j in range(0, n+1):
                    if used[j]: u[p[j]] += delta; v[j] -= delta
                    else: minv[j] -= delta
                j0 = j1
                if p[j0] == 0: break
            while True:
                j1 = way[j0]; p[j0] = p[j1]; j0 = j1
                if j0 == 0: break
        assign = np.zeros(n, dtype=int)
        for j in range(1, n+1): assign[j-1] = p[j]-1
        return assign

    assign = hungarian(pad)
    chosen_rows = assign[:n_cols]
    total = 0.0
    for j, i in enumerate(chosen_rows):
        if i < score_matrix.shape[0]: total += score_matrix[i, j]
    return chosen_rows.tolist(), total

# =========================
# Foxtrick-style effective skills & contributions (new)
# =========================

def ft_loyalty_bonus(loyalty: Optional[float], is_mother_club: bool) -> float:
    if is_mother_club:
        return 1.5
    if loyalty is None:
        return 0.0
    return max(0.0, loyalty - 1.0) / 19.0

def ft_average_energy_90(stamina_level: float) -> float:
    # stamina is 1-based float (HT stamina 1..10)
    s = float(stamina_level)
    if s >= 8.63:
        return 1.0
    total = 0.0
    initial = 1.0 + (0.0292 * s + 0.05)
    if s > 8:
        initial += 0.15 * (s - 8)
    decay = max(0.0325, -0.0039 * s + 0.0634)
    rest = 0.1875
    for chk in range(1, 19):  # 18 checkpoints
        cur = initial - chk * decay
        if chk > 9:
            cur += rest
        cur = min(1.0, cur)
        total += cur
    return total / 18.0

def ft_effective_skills(p: Player, use_exp: bool, use_loy: bool, use_sta: bool, use_form: bool, bruised: bool) -> Dict[str, float]:
    eff = {
        "keeper": float(p.KeeperSkill),
        "defending": float(p.DefenderSkill),
        "playmaking": float(p.PlaymakerSkill),
        "winger": float(p.WingerSkill),
        "passing": float(p.PassingSkill),
        "scoring": float(p.ScorerSkill),
    }
    # Experience bonus: log10(exp)*4/3 to all skills (if exp>0)
    if use_exp:
        try:
            ex = float(p.Experience)
            if ex > 0:
                bonus = math.log10(ex) * 4.0 / 3.0
                for k in eff:
                    eff[k] += bonus
        except:
            pass
    # Loyalty
    if use_loy:
        bonus = ft_loyalty_bonus(p.Loyalty, bool(p.HomeGrown))
        for k in eff:
            eff[k] += bonus
    # Stamina -> energy multiplier
    if use_sta:
        e = ft_average_energy_90(float(p.Stamina))
        for k in eff:
            eff[k] *= e
    # Form multiplier (index 0..8). Clamp range.
    if use_form:
        form_infls = [0, 0.305, 0.5, 0.629, 0.732, 0.82, 0.897, 0.967, 1]
        idx = int(round(float(p.Form)))
        idx = max(0, min(8, idx))
        factor = form_infls[idx]
        for k in eff:
            eff[k] *= factor
    # Bruised penalty
    if bruised:
        for k in eff:
            eff[k] *= 0.95
    return eff

def ft_contribution_factors(
    CTR_VS_WG=1.4, WBD_VS_CD=1.7, WO_VS_FW=1.3, IM_VS_CD=0.6, MF_VS_ATT=3.0, DF_VS_ATT=1.1
):
    factors = {
        "kp": {
            "keeper":   [0.87, 0.61, 0.61],
            "defending":[0.35, 0.25, 0.25],
        },
        "cd": {
            "defending":[1.00, 0.26, 0.26],
            "playmaking":0.25,
        },
        "wb": {
            "defending":[0.38, 0.92],
            "playmaking":0.15,
            "winger":   [0.00, 0.59],
        },
        "im": {
            "defending":[0.40, 0.09, 0.09],
            "playmaking":1.00,
            "passing":  [0.33, 0.13, 0.13],
            "scoring":  [0.22, 0.00],
        },
        "w": {
            "defending":[0.20, 0.35],
            "playmaking":0.45,
            "passing": [0.11, 0.26],
            "winger":  [0.00, 0.86],
        },
        "fw": {
            "playmaking":0.25,
            "passing": [0.33, 0.14, 0.14],
            "scoring": [1.00, 0.27, 0.27],
            "winger":  [0.00, 0.24, 0.24],
        },
    }

    def mk(skill, contr):
        is_def = (skill in ("defending", "keeper"))
        if isinstance(contr, list):
            center = float(contr[0])
            side   = float(contr[1]) if len(contr) > 1 else 0.0
            fars   = float(contr[2]) if len(contr) > 2 else 0.0
            side  *= (WBD_VS_CD if is_def else WO_VS_FW)
            fars  *= (WBD_VS_CD if is_def else WO_VS_FW)
            wings  = side + fars
            factor = (center + wings / CTR_VS_WG) / (1.0 + 2.0 / CTR_VS_WG)
            if is_def:
                factor *= DF_VS_ATT
            return {"center":center, "side":side, "farSide":fars, "wings":wings, "factor":factor}
        else:
            center = float(contr) * IM_VS_CD
            factor = center * MF_VS_ATT
            return {"center":center, "side":0.0, "farSide":0.0, "wings":0.0, "factor":factor}

    ret = {}
    for pos, skills in factors.items():
        ret[pos] = {skill: mk(skill, contr) for skill, contr in skills.items()}
    return ret

def ft_role_score(player: Player, pos_code: str,
                  use_exp: bool, use_loy: bool, use_sta: bool, use_form: bool, bruised: bool,
                  params=None) -> tuple[float, dict]:
    if params is None:
        params = dict(CTR_VS_WG=1.4, WBD_VS_CD=1.7, WO_VS_FW=1.3, IM_VS_CD=0.6, MF_VS_ATT=3.0, DF_VS_ATT=1.1)
    table = ft_contribution_factors(**params)
    eff = ft_effective_skills(player, use_exp, use_loy, use_sta, use_form, bruised)
    total = 0.0
    breakdown = {}
    for skill, desc in table[pos_code].items():
        key = "keeper" if skill == "keeper" else skill
        val = eff[key]
        part = val * desc["factor"]
        breakdown[skill] = part
        total += part
    return total, breakdown

# =========================
# UI
# =========================

PRESETS = {
    "3-5-2":  ["GK","CD","RB","LB","IM","IM","IM","RW","LW","FW","FW"],
    "5-3-2":  ["GK","CD","CD","CD","RB","LB","IM","RW","LW","FW","FW"],
    "4-4-2":  ["GK","CD","CD","RB","LB","IM","IM","RW","LW","FW","FW"],
    "4-3-3":  ["GK","CD","CD","RB","LB","IM","IM","IM","RW","LW","FW"],
    "3-4-3":  ["GK","CD","CD","CD","RB","LB","IM","IM","RW","LW","FW"],
    "4-2-3-1":["GK","CD","CD","RB","LB","IM","IM","RW","LW","FW","FW"],
}

st.set_page_config(page_title="NT Selector – Best XI + Foxtrick Top 10", layout="wide")
st.title("NT Selector – Best XI (HO) + Top 10 (Foxtrick)")

with st.sidebar:
    st.subheader("1) Upload NT_Prospects.csv")
    up = st.file_uploader("CSV", type=["csv"])
    st.caption("No Position needed; everyone is rated for each role.")

    st.subheader("2) Match context (HO engine)")
    location = st.selectbox("Location", ["HOME","AWAY","AWAY_DERBY"], 0)
    attitude = st.selectbox("Attitude", ["NORMAL","PIC","MOTS"], 0)
    tactic = st.selectbox("Tactic", ["None","AttackInTheMiddle","AttackInWings","PlayCreatively","CounterAttacks","LongShots","Pressing"], 0)
    weather = st.selectbox("Weather", ["Normal","RAINY","SUNNY"], 0)
    coach = st.slider("Coach modifier (−10 def … +10 off)", -10, 10, 0)
    ts = st.slider("Team Spirit", 0.0, 8.0, 7.0, 0.5)
    conf = st.slider("Confidence", 1, 9, 6)
    minute = st.slider("Minute", 0, 90, 45, step=5)
    gk_min = st.slider("GK min goalkeeping", 0, 20, 14)

    st.subheader("3) Filters")
    # Min filters (>=)
    min_age = st.number_input("Min Age (≥)", 0, 60, 0)
    min_form = st.slider("Min Form (≥)", 1, 8, 1)
    min_stamina = st.slider("Min Stamina (≥)", 1, 10, 1)
    min_xp = st.slider("Min Experience (≥)", 0, 20, 0)

    st.subheader("4) Formation")
    mode = st.radio("Mode", ["Presets","Custom"], index=0)
    if mode == "Presets":
        preset_name = st.selectbox("Choose formation", list(PRESETS.keys()), index=0)
        roles = PRESETS[preset_name][:]
    else:
        st.caption("Set how many of each role you want. Total outfielders must be 10.")
        n_cd = st.number_input("Central defenders (CD)", 0, 5, 2)
        n_rb = st.number_input("Right back / wingback (RB)", 0, 1, 1)
        n_lb = st.number_input("Left back / wingback (LB)", 0, 1, 1)
        n_im = st.number_input("Inner mids (IM)", 0, 5, 2)
        n_rw = st.number_input("Right wingers (RW)", 0, 2, 1)
        n_lw = st.number_input("Left wingers (LW)", 0, 2, 1)
        n_fw = st.number_input("Forwards (FW)", 0, 4, 2)
        total_outfield = n_cd + n_rb + n_lb + n_im + n_rw + n_lw + n_fw
        st.caption(f"Outfielders: **{total_outfield}/10**")
        roles = ["GK"] + ["CD"]*n_cd + ["RB"]*n_rb + ["LB"]*n_lb + \
                ["IM"]*n_im + ["RW"]*n_rw + ["LW"]*n_lw + ["FW"]*n_fw

    st.subheader("5) Foxtrick options")
    use_exp = st.checkbox("Use Experience bonus", value=True)
    use_loy = st.checkbox("Use Loyalty bonus", value=True)
    use_sta = st.checkbox("Use Stamina energy factor", value=True)
    use_form = st.checkbox("Use Form multiplier", value=True)
    use_bruised = st.checkbox("Bruised penalty (−5%)", value=False)

    st.caption("Foxtrick parameters")
    CTR_VS_WG = st.number_input("CTR_VS_WG", value=1.4, step=0.1, format="%.2f")
    WBD_VS_CD = st.number_input("WBD_VS_CD", value=1.7, step=0.1, format="%.2f")
    WO_VS_FW = st.number_input("WO_VS_FW", value=1.3, step=0.1, format="%.2f")
    IM_VS_CD = st.number_input("IM_VS_CD", value=0.6, step=0.05, format="%.2f")
    MF_VS_ATT = st.number_input("MF_VS_ATT", value=3.0, step=0.1, format="%.2f")
    DF_VS_ATT = st.number_input("DF_VS_ATT", value=1.1, step=0.1, format="%.2f")
    ft_params = dict(
        CTR_VS_WG=CTR_VS_WG, WBD_VS_CD=WBD_VS_CD, WO_VS_FW=WO_VS_FW,
        IM_VS_CD=IM_VS_CD, MF_VS_ATT=MF_VS_ATT, DF_VS_ATT=DF_VS_ATT
    )

if up is None:
    st.info("Upload NT_Prospects.csv to start.")
    st.stop()

players_all = load_players_from_csv(up)

# Apply filters
players: List[Player] = []
for p in players_all:
    age_ok = (p.Age is None) or (p.Age >= min_age)
    if not age_ok: continue
    if p.Form < min_form: continue
    if p.Stamina < min_stamina: continue
    try:
        xp_val = float(p.Experience)
    except:
        try: xp_val = float(str(p.Experience).strip())
        except: xp_val = 0.0
    if xp_val < min_xp: continue
    players.append(p)

st.caption(f"Filtered players: **{len(players)}/{len(players_all)}**")

# Validate formation
if roles[0] != "GK":
    roles = ["GK"] + roles  # safety
if len(roles) != 11:
    st.error("Your formation must have 11 players (1 GK + 10 outfielders).")
    st.stop()

# Build labels like IM1, IM2...
role_instances, counter = [], {}
for r in roles:
    c = counter.get(r, 0) + 1; counter[r] = c
    role_instances.append(f"{r}{c}" if roles.count(r) > 1 and r != "GK" else r)

# GK selection (avg of DC/DL/DR contributions)
def gk_score_triplet(pl: Player) -> Tuple[float,float,float,float]:
    dr = player_sector_bar_single(pl,"GK","M","DR",minute,tactic,weather,attitude,location,coach,ts,conf)
    dc = player_sector_bar_single(pl,"GK","M","DC",minute,tactic,weather,attitude,location,coach,ts,conf)
    dl = player_sector_bar_single(pl,"GK","M","DL",minute,tactic,weather,attitude,location,coach,ts,conf)
    return (dc+dl+dr)/3.0, dc, dl, dr

gk_cands = []
for idx, p in enumerate(players):
    if p.KeeperSkill >= gk_min:
        avg, dc, dl, dr = gk_score_triplet(p)
        gk_cands.append((idx, p.Name, avg, dc, dl, dr))
gk_df = pd.DataFrame(gk_cands, columns=["player_idx","Name","GK score (avg DC/DL/DR)","DC","DL","DR"])\
        .sort_values("GK score (avg DC/DL/DR)", ascending=False)

chosen_gk = int(gk_df.iloc[0]["player_idx"]) if "GK" in roles and not gk_df.empty else None

# Field roles (everything except GK)
field_roles = [r for r in role_instances if r != "GK"]

# Score matrix players x roles (HO engine)
score_cols = []
for r in field_roles:
    base = r.rstrip("0123456789")
    tr, side, tgt = ROLE_DEF[base]
    score_cols.append((r, tr, side, tgt))

nP, nR = len(players), len(score_cols)
M = np.zeros((nP, nR), dtype=float)
for i, p in enumerate(players):
    if chosen_gk is not None and i == chosen_gk:
        M[i,:] = -1e6
        continue
    for j, (_, tr, side, tgt) in enumerate(score_cols):
        M[i,j] = player_sector_bar_single(p, tr, side, tgt, minute, tactic, weather, attitude, location, coach, ts, conf)

assign_rows, total = hungarian_maximize(M)

# Build XI table
xi_rows = []
if "GK" in roles:
    if chosen_gk is None:
        xi_rows.append(("GK","(no GK ≥ threshold)","","","", "", "", ""))
    else:
        g = players[chosen_gk]; avg, dc, dl, dr = gk_score_triplet(g)
        xi_rows.append(("GK", g.Name, f"{avg:.2f}", f"DC {dc:.2f} | DL {dl:.2f} | DR {dr:.2f}",
                        g.Specialty, g.Experience, g.Form, g.Stamina))

selected_names = set()
for j, row_idx in enumerate(assign_rows):
    role_label = score_cols[j][0]
    if row_idx >= nP or M[row_idx,j] < 0:
        xi_rows.append((role_label,"(unfilled)","","","","","",""))
        continue
    name = players[row_idx].Name
    selected_names.add(name)
    p = players[row_idx]
    xi_rows.append((role_label, name, f"{M[row_idx,j]:.2f}","", p.Specialty, p.Experience, p.Form, p.Stamina))

xi_df = pd.DataFrame(xi_rows, columns=["Role","Player","Score","(GK breakdown)","Specialty","Experience","Form","Stamina"])

# Benches
bench_rows = []
if not gk_df.empty and len(gk_df) > 1:
    for _, row in gk_df.iloc[1:6].iterrows():
        bench_rows.append(("GK", row["Name"], f'{row["GK score (avg DC/DL/DR)"]:.2f}'))

present_bases = sorted(set([r.rstrip("0123456789") for r in field_roles]))
for base in present_bases:
    j = next(k for k,(nm,_,_,_) in enumerate(score_cols) if nm.rstrip("0123456789")==base)
    order = np.argsort(-M[:,j])
    cnt=0
    for i in order:
        if chosen_gk is not None and i == chosen_gk: continue
        n = players[i].Name
        if n in selected_names: continue
        bench_rows.append((base, n, f"{M[i,j]:.2f}"))
        cnt+=1
        if cnt>=5: break
bench_df = pd.DataFrame(bench_rows, columns=["Role","Player","Score"])

# Scouting (HO role scores)
scout = {"Name":[p.Name for p in players]}
for base,(tr,side,tgt) in ROLE_DEF.items():
    scout[base] = [player_sector_bar_single(p,tr,side,tgt,minute,tactic,weather,attitude,location,coach,ts,conf)
                   for p in players]
scout_df = pd.DataFrame(scout).sort_values("Name")

# =========================
# Tabs: Best XI (HO), Foxtrick Top10, Scouting
# =========================

tab_ho, tab_ft, tab_scout = st.tabs(["Best XI (HO)", "Foxtrick: Top 10 contributors", "Scouting (HO roles)"])

with tab_ho:
    title = next((k for k,v in PRESETS.items() if v == roles), None) if mode=="Presets" else "Custom"
    st.subheader(f"Best XI – {title}")
    st.dataframe(xi_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Bench suggestions")
        st.dataframe(bench_df, use_container_width=True)
    with c2:
        st.subheader("GK candidates (≥ threshold)")
        if not gk_df.empty:
            st.dataframe(
                gk_df.drop(columns=["player_idx"]),
                use_container_width=True
            )
        else:
            st.info("No GK meets threshold.")

    st.download_button("Download XI (CSV)", xi_df.to_csv(index=False).encode("utf-8"), file_name="best_xi.csv")
    st.download_button("Download Bench (CSV)", bench_df.to_csv(index=False).encode("utf-8"), file_name="bench.csv")

with tab_ft:
    st.subheader("Top 10 by role – Foxtrick contributions")

    def build_top10(players, label, code):
        rows = []
        for p in players:
            if code == "kp" and float(p.KeeperSkill) < float(gk_min):
                continue
            score, parts = ft_role_score(p, code, use_exp, use_loy, use_sta, use_form, use_bruised, ft_params)
            rows.append({
                "Name": p.Name,
                "Side": (p.Side or ""),
                "Specialty": (p.Specialty or "None"),
                "Experience": p.Experience,
                "Form": p.Form,
                "Stamina": p.Stamina,
                "Contribution": score,
            })
        if not rows:
            return label, pd.DataFrame(columns=["Name","Side","Specialty","Experience","Form","Stamina","Contribution"])
        df = pd.DataFrame(rows).sort_values("Contribution", ascending=False).head(10).reset_index(drop=True)
        return label, df

    role_defs = [
        ("GK",  "kp"),
        ("CD",  "cd"),
        ("WBR", "wb"),
        ("WBL", "wb"),
        ("IM",  "im"),
        ("WGR", "w"),
        ("WGL", "w"),
        ("FW",  "fw"),
    ]

    grid = [role_defs[0:4], role_defs[4:8]]
    for row in grid:
        cols = st.columns(4)
        for (col, (label, code)) in zip(cols, row):
            with col:
                name, df = build_top10(players, label, code)
                st.markdown(f"**{name}**")
                if df.empty:
                    st.info("empty")
                else:
                    st.dataframe(df[["Name","Side","Specialty","Experience","Form","Stamina","Contribution"]]
                                 .style.format({"Contribution":"{:.2f}"}), use_container_width=True)
                    st.download_button(
                        f"Download {name} Top 10",
                        df.to_csv(index=False).encode("utf-8"),
                        file_name=f"top10_{name.lower()}.csv",
                        use_container_width=True,
                    )

with tab_scout:
    st.subheader("Per-player scouting table (HT-scaled scores by role)")
    st.dataframe(scout_df, use_container_width=True)
    st.download_button("Download Scouting (CSV)", scout_df.to_csv(index=False).encode("utf-8"), file_name="scouting_roles.csv")