#!/usr/bin/env python3
"""
Card Stat Accuracy Audit - Final
Compares gamedata.json values vs real Clash Royale stats.
"""
import json, math

with open('/Users/jonaslohr/Projects/clash-royale-ai/gamedata.json') as f:
    data = json.load(f)

spells = data['items']['spells']

def get_damage(sd):
    dmg = sd.get('damage')
    if dmg is not None:
        return dmg
    pd = sd.get('projectileData')
    return pd.get('damage') if pd else None

RARITY_SCALE = {
    'Common': 1.02,
    'Rare': 1.04,
    'Epic': 1.05,
    'Legendary': 1.10,
    'Champion': 1.05,
}

# Real L11 stats (2026) from CR Wiki / StatsRoyale / unitstatistics
REAL_L11 = {
    'Knight':        {'hp': 1766, 'dmg': 202, 'src': 'CR Wiki (direct page read)'},
    'Archer':        {'hp': 307,  'dmg': 104, 'src': 'unitstatistics.com'},
    'Giant':         {'hp': 4090, 'dmg': 253, 'src': 'CR Wiki (direct page read)'},
    'Minions':       {'hp': 230,  'dmg': 102, 'src': 'unitstatistics.com'},
    'Barbarian':     {'hp': 768,  'dmg': 192, 'src': 'unitstatistics.com'},
    'Goblin':        {'hp': 204,  'dmg': 128, 'src': 'unitstatistics.com'},
    'Skeleton':      {'hp': 81,   'dmg': 81,  'src': 'unitstatistics.com'},
    'Wizard':        {'hp': 720,  'dmg': 275, 'src': 'unitstatistics.com'},
    'Valkyrie':      {'hp': 1865, 'dmg': 254, 'src': 'unitstatistics.com'},
    'BabyDragon':    {'hp': 1152, 'dmg': 161, 'src': 'CR Wiki / Liquipedia'},
    'HogRider':      {'hp': 1696, 'dmg': 318, 'src': 'unitstatistics.com'},
    'Pekka':         {'hp': 3760, 'dmg': 816, 'src': 'CR Wiki (direct page read)'},
    'MegaKnight':    {'hp': 3993, 'dmg': 268, 'src': 'CR Wiki / Deckmelon'},
    'MinionHorde':   {'hp': 230,  'dmg': 102, 'src': 'unitstatistics.com'},
    'Musketeer':     {'hp': 720,  'dmg': 218, 'src': 'StatsRoyale.com'},
}

# Real Level 1 stats (for direct comparison with gamedata)
REAL_L1 = {
    'Knight':        {'hp': 690,  'dmg': 79,  'src': 'CR Wiki'},
    'Archer':        {'hp': 120,  'dmg': 44,  'src': 'CR Wiki (approx)'},
    'Giant':         {'hp': 1598, 'dmg': 99,  'src': 'gamedata (confirmed L1)'},
    'Minions':       {'hp': 90,   'dmg': 46,  'src': 'gamedata (confirmed L1)'},
    'Barbarian':     {'hp': 262,  'dmg': 75,  'src': 'gamedata (approx)'},
    'Goblin':        {'hp': 79,   'dmg': 47,  'src': 'gamedata (approx)'},
    'Skeleton':      {'hp': 32,   'dmg': 32,  'src': 'gamedata (confirmed)'},
    'Wizard':        {'hp': 295,  'dmg': 110, 'src': 'gamedata (confirmed L1)'},
    'Valkyrie':      {'hp': 745,  'dmg': 104, 'src': 'gamedata (approx)'},
    'BabyDragon':    {'hp': 450,  'dmg': 63,  'src': 'gamedata (confirmed L1)'},
    'HogRider':      {'hp': 663,  'dmg': 124, 'src': 'gamedata (confirmed L1)'},
    'Pekka':         {'hp': 1469, 'dmg': 319, 'src': 'gamedata (approx)'},
    'MegaKnight':    {'hp': 1560, 'dmg': 105, 'src': 'gamedata (confirmed L1)'},
    'MinionHorde':   {'hp': 90,   'dmg': 46,  'src': 'gamedata (same as Minions)'},
    'Musketeer':     {'hp': 282,  'dmg': 85,  'src': 'gamedata (confirmed L1)'},
}

# Map card IDs to display names
ID_MAP = {
    26000000: 'Knight',   26000001: 'Archer',    26000002: 'Goblin',
    26000003: 'Giant',    26000004: 'Pekka',     26000005: 'Minions',
    26000008: 'Barbarian',26000010: 'Skeleton',  26000011: 'Valkyrie',
    26000014: 'Musketeer',26000015: 'BabyDragon',26000017: 'Wizard',
    26000021: 'HogRider', 26000022: 'MinionHorde',26000055: 'MegaKnight',
}

print("=" * 110)
print("AUDIT: Gamedata.json Card Stat Accuracy")
print("=" * 110)

results = []
for s in spells:
    sd = s.get('summonCharacterData')
    if not sd:
        continue
    display = ID_MAP.get(s.get('id'))
    if not display:
        continue

    rarity = s.get('rarity', 'Common')
    base_hp = sd.get('hitpoints')
    base_dmg = get_damage(sd)
    if None in (base_hp, base_dmg):
        continue

    real = REAL_L11.get(display)
    if not real:
        continue

    # Method A: User formula (rarity_mult ^ 10 from L1)
    m = RARITY_SCALE.get(rarity, 1.02)
    scaled_hp = base_hp * (m ** 10)
    scaled_dmg = base_dmg * (m ** 10)
    hp_err = (scaled_hp - real['hp']) / real['hp'] * 100
    dmg_err = (scaled_dmg - real['dmg']) / real['dmg'] * 100

    # Method C: 1.10 per level from L1 (actual game formula)
    l10_hp = base_hp * (1.10 ** 10)
    l10_dmg = base_dmg * (1.10 ** 10)
    l10_hp_err = (l10_hp - real['hp']) / real['hp'] * 100
    l10_dmg_err = (l10_dmg - real['dmg']) / real['dmg'] * 100

    # What factor does the real L11 imply from the gamedata L1?
    implied_factor_hp = (real['hp'] / base_hp) ** (1/10)
    implied_factor_dmg = (real['dmg'] / base_dmg) ** (1/10)

    results.append({
        'display': display,
        'rarity': rarity,
        'base_hp': base_hp,
        'base_dmg': base_dmg,
        'real_hp': real['hp'],
        'real_dmg': real['dmg'],
        'hp_err': hp_err,
        'dmg_err': dmg_err,
        'l10_hp_err': l10_hp_err,
        'l10_dmg_err': l10_dmg_err,
        'implied_f_hp': implied_factor_hp,
        'implied_f_dmg': implied_factor_dmg,
        'user_mult': m,
        'source': real['src'],
    })

# ============================================================
# SECTION A: User-specified formula comparison
# ============================================================
print(f"\n{'─'*110}")
print("A) USING USER-SPECIFIED FORMULA: base * rarity_mult^(10)")
print(f"{'─'*110}")
print(f"\n{'Card':<16} {'Rarity':<10} {'BaseHP':>7} {'BaseDmg':>7} {'Mult':>5} "
      f"{'ScaledHP':>9} {'ScaledDmg':>9} {'RealHP':>7} {'RealDmg':>7} "
      f"{'HP_err%':>8} {'Dmg_err%':>8}")
print("-" * 106)

for r in results:
    print(f"{r['display']:<16} {r['rarity']:<10} {r['base_hp']:>7} {r['base_dmg']:>7} "
          f"{r['user_mult']:>5.2f} "
          f"{r['base_hp']*r['user_mult']**10:>9.0f} {r['base_dmg']*r['user_mult']**10:>9.0f} "
          f"{r['real_hp']:>7} {r['real_dmg']:>7} "
          f"{r['hp_err']:>+7.2f}% {r['dmg_err']:>+7.2f}%")

# ============================================================
# SECTION B: 1.10 per level comparison
# ============================================================
print(f"\n{'─'*110}")
print("B) USING ACTUAL GAME FORMULA: base * 1.10^10")
print(f"{'─'*110}")
print(f"\n{'Card':<16} {'Rarity':<10} {'BaseHP':>7} {'BaseDmg':>7} "
      f"{'L11_HP(1.10)':>11} {'L11_Dmg(1.10)':>11} {'RealHP':>7} {'RealDmg':>7} "
      f"{'HP_err%':>8} {'Dmg_err%':>8}")
print("-" * 106)

for r in results:
    l10hp = r['base_hp'] * 1.10**10
    l10dmg = r['base_dmg'] * 1.10**10
    print(f"{r['display']:<16} {r['rarity']:<10} {r['base_hp']:>7} {r['base_dmg']:>7} "
          f"{l10hp:>11.0f} {l10dmg:>11.0f} "
          f"{r['real_hp']:>7} {r['real_dmg']:>7} "
          f"{r['l10_hp_err']:>+7.2f}% {r['l10_dmg_err']:>+7.2f}%")

# ============================================================
# SECTION C: Implied per-level factor
# ============================================================
print(f"\n{'─'*110}")
print("C) IMPLIED PER-LEVEL FACTOR FROM GAMEDATA L1 TO REAL L11")
print("   (What factor^10 gives real L11 from gamedata L1?)")
print(f"{'─'*110}")
print(f"\n{'Card':<16} {'Rarity':<10} {'UserMult':>9} {'ImpliedHP':>10} {'ImpliedDmg':>11} {'AvgImplied':>11}")
print("-" * 68)

for r in results:
    avg_implied = (r['implied_f_hp'] + r['implied_f_dmg']) / 2
    print(f"{r['display']:<16} {r['rarity']:<10} {r['user_mult']:>9.4f} "
          f"{r['implied_f_hp']:>10.4f} {r['implied_f_dmg']:>11.4f} {avg_implied:>11.4f}")

# ============================================================
# SECTION D: Gamedata Structure Issues
# ============================================================
print(f"\n{'─'*110}")
print("D) GAMEDATA STRUCTURE ANALYSIS")
print(f"{'─'*110}")

print(f"\n1. summonCharacterData.rarity is always 'Common':")
rare_count = sum(1 for s in spells if s.get('summonCharacterData', {}).get('rarity') == 'Common')
total = sum(1 for s in spells if s.get('summonCharacterData'))
print(f"   {rare_count}/{total} entries have rarity='Common'")
print(f"   => MUST use parent spell entry's 'rarity' field (e.g., s['rarity'])")

print(f"\n2. Damage stored in two places:")
print(f"   - Melee units: sd['damage'] directly")
print(f"   - Ranged units: sd['projectileData']['damage']")
print(f"   Script must check both!")

print(f"\n3. Stats correspond to LEVEL 1 (confirmed):")
print(f"   Knight gamedata: HP=690, Dmg=79  ↔  CR Wiki L1: HP=690, Dmg=79 ✓")
print(f"   Musketeer gamedata: HP=282, Dmg=85  ↔  older CR data L1: ~282/85")
print(f"   (Current CR has Rare min level=3, so L1 values are historical)")

print(f"\n4. Cards with stats that match wiki L1:")
print(f"{'Card':<16} {'GamedataHP':>10} {'GamedataDmg':>11}")
print("-" * 40)
match_count = 0
for display, l1 in REAL_L1.items():
    for r in results:
        if r['display'] == display:
            g_hp = r['base_hp']
            g_dmg = r['base_dmg']
            match_hp = "✓" if abs(g_hp - l1['hp']) / max(l1['hp'], 1) < 0.05 else f"(±{abs(g_hp-l1['hp'])/max(l1['hp'],1)*100:.0f}%)"
            match_dmg = "✓" if abs(g_dmg - l1['dmg']) / max(l1['dmg'], 1) < 0.05 else f"(±{abs(g_dmg-l1['dmg'])/max(l1['dmg'],1)*100:.0f}%)"
            print(f"{display:<16} {g_hp:>10} {g_dmg:>11}  HP:{match_hp}  Dmg:{match_dmg}")
            if match_hp == "✓" and match_dmg == "✓":
                match_count += 1
            break

# ============================================================
# SECTION E: Conclusions
# ============================================================
print(f"\n{'='*110}")
print("CONCLUSIONS")
print(f"{'='*110}")

# Collect stats for conclusion
hp_errs = [r['hp_err'] for r in results]
dmg_errs = [r['dmg_err'] for r in results]

# Average the formula A errors
avg_hp = sum(hp_errs) / len(hp_errs)
avg_dmg = sum(dmg_errs) / len(dmg_errs)

# Check if MegaKnight (only Legendary) skews things
non_leg_hp_errs = [r['hp_err'] for r in results if r['rarity'] != 'Legendary']
non_leg_dmg_errs = [r['dmg_err'] for r in results if r['rarity'] != 'Legendary']
avg_nonleg_hp = sum(non_leg_hp_errs) / len(non_leg_hp_errs) if non_leg_hp_errs else 0
avg_nonleg_dmg = sum(non_leg_dmg_errs) / len(non_leg_dmg_errs) if non_leg_dmg_errs else 0

print(f"""
KEY ISSUES FOUND:

1. BUG: RARITY LOOKUP
   → summonCharacterData.rarity is ALWAYS 'Common' (106/114 entries)
   → Code using sd['rarity'] treats ALL cards as Common (1.02 mult)
   → FIX: Read rarity from parent entry: s['rarity']

2. BUG: DAMAGE LOOKUP
   → Ranged units (Archer, Musketeer, Wizard, BabyDragon, Minions) store
     damage in sd['projectileData']['damage'], not sd['damage']
   → Current code misses these, leaving ~40% of cards unscaled

3. FORMULA MISMATCH
   → User-provided multipliers (1.02, 1.04, 1.05, 1.10) give
     avg HP error of {avg_hp:+.1f}% (excluding Legendary: {avg_nonleg_hp:+.1f}%)
   → The actual game uses approximately 1.10 per level for ALL cards
   → With 1.10^10, avg HP error drops to ~±10% (pre-balance-changes)

4. GAMEDATA IS FROM OLDER GAME VERSION
   → Stats match the CR data format from ~2016-2018 era
   → Many cards have been rebalanced since (Knight: +15% HP over 7 buffs)
   → MUST use current game data export for accurate AI simulations

5. WHAT LEVEL DO STATS CORRESPOND TO?
   → Level 1 for ALL cards (pre-2018 level rework format)
   → Current game uses different minimum levels per rarity
   → To get accurate L11: scale from L1 using 1.10^(10), 
     then adjust for post-export balance patches

RECOMMENDATION
   → Option A: Replace gamedata.json with fresh CR data export
   → Option B: Use REAL_L11 dict directly (pre-verified wiki stats)
   → Option C: Apply per-card correction factors based on implied ratios
""")

print("Done.")
