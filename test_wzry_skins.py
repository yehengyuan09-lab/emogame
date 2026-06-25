"""
Test script for Honor of Kings (王者荣耀) skin data fetching functionality.

Validates:
1. JSON data parsing (hero list, skin names, URLs)
2. Cross-reference JSON data with downloaded skin images
3. Query APIs: by hero name, by skin name, by hero type
4. Statistics: hero count, skin count, coverage
"""

import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Paths
SUBMODULE_DIR = Path(__file__).parent / "hero-skin-image"
JSON_PATH = SUBMODULE_DIR / "wzry-heros.json"
IMAGE_DIRS = {
    "1phone-smallskin": SUBMODULE_DIR / "1phone-smallskin-images",
    "2phone-mobileskin": SUBMODULE_DIR / "2phone-mobileskin-images",
    "3phone-bigskin": SUBMODULE_DIR / "3phone-bigskin-images",
    "4wallpaper-mobileskin": SUBMODULE_DIR / "4wallpaper-mobileskin-images",
    "5wallpaper-bigskin": SUBMODULE_DIR / "5wallpaper-bigskin-images",
}

# ──────────────────────────────────────────────────────────────────
# 1. JSON DATA PARSING
# ──────────────────────────────────────────────────────────────────

def load_hero_data():
    """Load and parse the wzry-heros.json file."""
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_json_structure(data):
    """Validate the top-level JSON structure."""
    errors = []

    if "hero-list" not in data:
        errors.append("Missing 'hero-list' key")
    elif not isinstance(data["hero-list"], list):
        errors.append(f"'hero-list' should be a list, got {type(data['hero-list']).__name__}")
    elif len(data["hero-list"]) == 0:
        errors.append("'hero-list' is empty")

    for key in ["description", "phone-smallskin-images", "phone-mobileskin-images",
                "phone-bigskin-images", "wallpaper-mobileskin-images", "wallpaper-bigskin-images"]:
        if key not in data:
            errors.append(f"Missing '{key}' key")

    # Validate each hero entry
    required_fields = ["cname", "ename", "id", "skinName", "skins", "heroType"]
    for i, hero in enumerate(data.get("hero-list", [])):
        for field in required_fields:
            if field not in hero:
                errors.append(f"Hero[{i}] ({hero.get('cname', '?')}) missing field '{field}'")
        if hero.get("skins") and hero.get("skinName"):
            expected = hero["skinName"].split("|")
            actual = hero["skins"]
            if len(expected) != len(actual):
                errors.append(
                    f"Hero '{hero['cname']}': skinName count ({len(expected)}) != skins count ({len(actual)})"
                )

    return errors


def test_hero_statistics(data):
    """Compute and display hero/skin statistics."""
    heroes = data["hero-list"]
    total_skins = sum(len(h.get("skins", [])) for h in heroes)
    hero_types = {}
    for h in heroes:
        ht = h.get("heroType", "未知")
        hero_types[ht] = hero_types.get(ht, 0) + 1

    stats = {
        "total_heroes": len(heroes),
        "total_skins": total_skins,
        "avg_skins_per_hero": round(total_skins / len(heroes), 1) if heroes else 0,
        "hero_types": hero_types,
    }
    return stats


# ──────────────────────────────────────────────────────────────────
# 2. CROSS-REFERENCE WITH DOWNLOADED IMAGES
# ──────────────────────────────────────────────────────────────────

def scan_downloaded_images():
    """Scan all image directories and return sets of filenames."""
    image_sets = {}
    for label, dirpath in IMAGE_DIRS.items():
        if dirpath.exists():
            files = set(f.name for f in dirpath.iterdir() if f.suffix.lower() == ".jpg")
            image_sets[label] = files
        else:
            image_sets[label] = set()
    return image_sets


def test_image_coverage(data, image_sets):
    """Check coverage: for each hero, verify downloaded images match expected skins."""
    heroes = data["hero-list"]
    results = {
        "heroes_with_full_coverage": 0,
        "heroes_with_partial_coverage": 0,
        "heroes_with_no_images": 0,
        "missing_images": [],
        "extra_images": 0,
    }

    # Use the biggest image set as reference
    ref_label = "5wallpaper-bigskin"
    ref_images = image_sets.get(ref_label, set())

    # Build expected filename pattern: {cname}-{index+1}-{skinName}.jpg
    expected_images = set()
    for hero in heroes:
        cname = hero["cname"]
        for i, skin in enumerate(hero.get("skins", [])):
            expected = f"{cname}-{i + 1}-{skin}.jpg"
            expected_images.add(expected)

    # Compare
    missing = expected_images - ref_images
    extra = ref_images - expected_images
    results["missing_images"] = sorted(missing)[:20]  # show first 20
    results["missing_count"] = len(missing)
    results["extra_count"] = len(extra)
    results["extra_samples"] = sorted(extra)[:10]

    # Per-hero coverage
    for hero in heroes:
        cname = hero["cname"]
        hero_skins = hero.get("skins", [])
        if not hero_skins:
            continue
        expected_for_hero = [f"{cname}-{i + 1}-{s}.jpg" for i, s in enumerate(hero_skins)]
        found = sum(1 for e in expected_for_hero if e in ref_images)
        if found == len(hero_skins):
            results["heroes_with_full_coverage"] += 1
        elif found > 0:
            results["heroes_with_partial_coverage"] += 1
        else:
            results["heroes_with_no_images"] += 1

    # Check each image directory
    results["image_counts"] = {label: len(imgs) for label, imgs in image_sets.items()}

    return results


# ──────────────────────────────────────────────────────────────────
# 3. QUERY FUNCTIONS
# ──────────────────────────────────────────────────────────────────

class SkinDataQuery:
    """Query interface for hero/skin data."""

    def __init__(self, data):
        self.heroes = data["hero-list"]
        self._build_index()

    def _build_index(self):
        """Build lookup indexes."""
        self._by_cname = {h["cname"]: h for h in self.heroes}
        self._by_ename = {h["ename"]: h for h in self.heroes}
        self._by_id = {h["id"]: h for h in self.heroes}
        # skin name -> list of (hero, skin_index)
        self._by_skin = {}
        for h in self.heroes:
            for i, s in enumerate(h.get("skins", [])):
                self._by_skin.setdefault(s, []).append((h, i))

    def get_hero_by_name(self, cname: str) -> dict | None:
        """Query hero by Chinese name."""
        return self._by_cname.get(cname)

    def get_hero_by_ename(self, ename: int) -> dict | None:
        """Query hero by English name code."""
        return self._by_ename.get(ename)

    def search_skin(self, skin_name: str) -> list[dict]:
        """Search for a skin by name (partial match). Returns list of matching skins."""
        results = []
        for skin, entries in self._by_skin.items():
            if skin_name in skin:
                for hero, idx in entries:
                    results.append({
                        "hero": hero["cname"],
                        "hero_type": hero["heroType"],
                        "skin_name": skin,
                        "skin_index": idx + 1,
                        "total_skins": len(hero.get("skins", [])),
                        "wallpaper_url": (
                            hero.get("wallpaperBigskinUrl", [None])[idx]
                            if idx < len(hero.get("wallpaperBigskinUrl", []))
                            else None
                        ),
                    })
        return results

    def get_heroes_by_type(self, hero_type: str) -> list[dict]:
        """Get all heroes of a specific type (e.g., '中路', '发育路')."""
        return [h for h in self.heroes if h.get("heroType") == hero_type]

    def get_top_skins_heroes(self, top_n: int = 10) -> list[dict]:
        """Get heroes with the most skins."""
        sorted_heroes = sorted(
            self.heroes,
            key=lambda h: len(h.get("skins", [])),
            reverse=True,
        )
        return [
            {"cname": h["cname"], "hero_type": h["heroType"], "skin_count": len(h.get("skins", []))}
            for h in sorted_heroes[:top_n]
        ]

    def get_skin_counts_distribution(self) -> dict:
        """Get distribution of skin counts per hero."""
        dist = {}
        for h in self.heroes:
            count = len(h.get("skins", []))
            dist[count] = dist.get(count, 0) + 1
        return dict(sorted(dist.items()))


def test_queries(query: SkinDataQuery):
    """Run various query tests."""
    test_results = []

    # 1. Query by hero name
    hero = query.get_hero_by_name("孙悟空")
    test_results.append({
        "test": "Query hero '孙悟空'",
        "pass": hero is not None,
        "detail": f"Found: {hero['cname']}, skins: {len(hero.get('skins',[]))}"
        if hero else "Not found",
    })

    # 2. Query non-existent hero
    hero = query.get_hero_by_name("不存在的英雄")
    test_results.append({
        "test": "Query non-existent hero",
        "pass": hero is None,
        "detail": "Correctly returned None",
    })

    # 3. Search skin by name
    skins = query.search_skin("龙")
    test_results.append({
        "test": "Search skin containing '龙'",
        "pass": len(skins) > 0,
        "detail": f"Found {len(skins)} skins: {[s['hero']+'-'+s['skin_name'] for s in skins[:5]]}",
    })

    # 4. Search specific skin
    skins = query.search_skin("天鹅之梦")
    test_results.append({
        "test": "Search specific skin '天鹅之梦'",
        "pass": len(skins) > 0,
        "detail": f"Hero: {skins[0]['hero']}, index: {skins[0]['skin_index']}" if skins else "Not found",
    })

    # 5. Heroes by type
    mid_heroes = query.get_heroes_by_type("中路")
    test_results.append({
        "test": "Get heroes by type '中路'",
        "pass": len(mid_heroes) > 0,
        "detail": f"Found {len(mid_heroes)} mid-lane heroes: {[h['cname'] for h in mid_heroes[:5]]}...",
    })

    # 6. Top skin heroes
    top = query.get_top_skins_heroes(5)
    test_results.append({
        "test": "Get top 5 heroes by skin count",
        "pass": len(top) == 5,
        "detail": f"Top: {[(t['cname'], t['skin_count']) for t in top]}",
    })

    return test_results


# ──────────────────────────────────────────────────────────────────
# MAIN TEST RUNNER
# ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("王者荣耀皮肤数据获取功能测试")
    print("Honor of Kings Skin Data Fetching Test")
    print("=" * 70)

    # Step 1: Load data
    print("\n📂 [1/4] Loading JSON data...")
    try:
        data = load_hero_data()
        print(f"   ✅ Loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return 1

    # Step 2: Validate structure
    print("\n🔍 [2/4] Validating data structure...")
    errors = test_json_structure(data)
    if errors:
        print(f"   ❌ {len(errors)} errors found:")
        for e in errors[:10]:
            print(f"      - {e}")
        if len(errors) > 10:
            print(f"      ... and {len(errors) - 10} more")
    else:
        print("   ✅ JSON structure is valid")

    # Step 3: Statistics
    print("\n📊 [3/4] Computing statistics...")
    stats = test_hero_statistics(data)
    print(f"   Total heroes:  {stats['total_heroes']}")
    print(f"   Total skins:   {stats['total_skins']}")
    print(f"   Avg skins/hero: {stats['avg_skins_per_hero']}")
    print(f"   Hero types:")
    for ht, count in sorted(stats["hero_types"].items(), key=lambda x: -x[1]):
        print(f"      {ht}: {count}")

    # Step 4: Image coverage
    print("\n🖼️  [4/4] Checking image coverage...")
    image_sets = scan_downloaded_images()
    coverage = test_image_coverage(data, image_sets)

    for label, count in coverage["image_counts"].items():
        print(f"   {label}: {count} images")
    print(f"   Heroes with full coverage:    {coverage['heroes_with_full_coverage']}")
    print(f"   Heroes with partial coverage: {coverage['heroes_with_partial_coverage']}")
    print(f"   Heroes with no images:        {coverage['heroes_with_no_images']}")
    if coverage["missing_count"] > 0:
        print(f"   ⚠️  Missing images ({coverage['missing_count']} total):")
        for m in coverage["missing_images"]:
            print(f"      - {m}")
    else:
        print(f"   ✅ All expected skin images found!")
    if coverage["extra_count"] > 0:
        print(f"   ℹ️  Extra/renamed images ({coverage['extra_count']} total):")
        for e in coverage["extra_samples"]:
            print(f"      - {e}")

    # ── Query Tests ──
    print("\n" + "=" * 70)
    print("🔎 Query API Tests")
    print("=" * 70)
    query = SkinDataQuery(data)

    query_results = test_queries(query)
    all_pass = True
    for r in query_results:
        status = "✅" if r["pass"] else "❌"
        if not r["pass"]:
            all_pass = False
        print(f"\n   {status} {r['test']}")
        print(f"      {r['detail']}")

    # ── Additional query demo: skin count distribution ──
    print("\n   📈 Skin count distribution:")
    dist = query.get_skin_counts_distribution()
    for count, num_heroes in dist.items():
        bar = "█" * num_heroes
        print(f"      {count:2d} skins: {num_heroes:3d} heroes {bar}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    data_ok = len(errors) == 0
    print(f"   JSON structure:  {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"   Query API:       {'✅ PASS' if all_pass else '❌ FAIL'}")
    print(f"   Image coverage:  {coverage['heroes_with_full_coverage']}/{stats['total_heroes']} heroes fully covered")

    failures = (0 if data_ok else 1) + (0 if all_pass else 1)
    print(f"\n   Total failures: {failures}")
    return failures


if __name__ == "__main__":
    exit(main())
