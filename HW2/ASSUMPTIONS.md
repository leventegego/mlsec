# HW2 Feltételezések és döntések

Ez a fájl nyilvántartja azokat a feltételezéseket és implementációs döntéseket, amelyeket a feladatleírás kétértelműségei miatt kellett meghozni.

---

## Előkészítés

| # | Téma | Feltételezés | Indoklás |
|---|---|---|---|
| P1 | Surrogate preprocessing | A surrogate modellek ugyanazon az előfeldolgozási pipeline-on mennek át, mint a clean modell (one-hot + StandardScaler fit a train adaton) | Logikai konzisztencia; a spec nem mondja ki explicit |
| P2 | Surrogate tanítóadat | A surrogateek a **teljes** tesztadaton tanulnak, validáció nélkül | A spec csak "tesztadat"-ot mond, részhalmazt nem említ |
| P3 | Surrogate label | A surrogate modellek bináris (0/1) címkékkel tanulnak, ugyanúgy mint a clean modell | A spec azt mondja "az első házi feladatban leírt módon" |

---

## 1. feladat: Untargeted Poisoning

| # | Téma | Feltételezés | Indoklás |
|---|---|---|---|
| U1 | Poison hozzáadás | Az invertált minták **hozzáadódnak** a tanítóadathoz, az eredetik bent maradnak (nem felülírás) | A spec explicit: "hozzáadja ezeket invertált címkével a tanítóadathoz" |
| U2 | Q1 — 5 futás | Mindegyik futás **különböző** véletlenszerűen kiválasztott mintatételt is jelent (nem csak különböző model seed) | Q1 lényege a random kiválasztás variabilitása |
| U3 | Q2 — 5 futás | A loss-based kiválasztás **determinisztikus** (same selected samples minden futásban); az 5 futás csak a modell random inicializációjában tér el | A loss ranking nem változik futások között |
| U4 | Q2 — loss számítás | A surrogate modellek (tesztadaton tanítva) loss értékét a **tanítóadat** mintáin számítjuk | A spec azt mondja "minden tanító minta loss értékét kiszámolja ... a surrogate modell mindegyikén" |
| U5 | Poison méret | p% a tanítóadat (80%-os split) méretéhez képest értendő, nem a teljes adathoz | "tanítóadat" = a 80%-os rész a spec szerint |

---

## 2. feladat: Targeted Poisoning (Witches' Brew)

| # | Téma | Feltételezés | Indoklás |
|---|---|---|---|
| T1 | WiB induló LR | Kezdeti learning rate: **0.1** (SGD optimizer-rel) | A spec nem adja meg; `ReduceLROnPlateau(patience=75)` hatékonyan csökkenti, 0.1 jó kiindulópont |
| T2 | Értékelés ismétlése | Targeted attack értékelése **egyszer** történik (egy retraining per target per p) | A spec nem ír ismétlésről; "a modell ... sorolja" egyes szám |
| T3 | Base minta pool | Base minták csak $y_i = 0$ (normal) tesztmintákból kerülnek ki; a 10 target ($y=1$) automatikusan kizárt | A spec: "y_adv_t = y_i", target-ek attack osztályúak tehát kizártak |
| T4 | Feature clipping | A [feat_min, feat_max] korlátokat a **skalázott** (StandardScaler utáni) train adaton számítjuk | A spec "skálázott feature értékek"-et említ |
| T5 | Poison per target | Minden target mintához **független** base minta kiválasztás és WiB optimalizáció fut | A spec "minden egyes target mintára" kitétele |
| T6 | Poison a train adathoz | A p poison mintát az eredeti 80%-os train adathoz adjuk (az untargeted poison nélkül, clean baseline) | Minden target-et külön értékelünk, nem kumulálódnak |
| T7 | WiB formula | Per-minta koszinusz-hasonlóság összege (nem összesített gradiens), ahogy a spec explicit kimondja | A spec megjegyzi, hogy ez pontosabb az előadás formulájánál |
| T8 | create_graph | A $\delta_i$ szerinti gradiens számításhoz `create_graph=True` szükséges (másodfokú optimalizáció) | $\delta_i \to \nabla_\theta \ell \to \cos\angle \to \nabla_{\delta_i}$ lánc |

---

## Implementációs döntések (kód-szintű)

| # | Téma | Döntés |
|---|---|---|
| I1 | `get_param_grad` detach | `create_graph=False` esetén detachol (base selection), `True` esetén nem (WiB PGD — kell a delta-ra áramló gradiens) |
| I2 | Model param grad nullázás | WiB PGD minden iteráció végén `model.zero_grad()` + `requires_grad_(False)` a surrogate modelleken, hogy ne halmozódjanak |
| I3 | Base minta precompute | `select_base_samples` max p=15-tel fut egyszer per target, majd p=5-nél az első 5 (leginkább hasonló) kerül kiválasztásra |
| I4 | `total_loss` inicializáció | `torch.zeros(1, device)[0]` — differenciálható nulla a gráfban |
| I5 | Best delta tracking | Minden iterációban ha a loss javul, elmentjük a delta-kat; a legjobb delta-t használjuk a poison minták összeállításához |
| I6 | Surrogate seed | Surrogate i: seed = `i * 1000 + 42` — garantáltan különböző random inicializáció minden surrogate-nek |
| I7 | Q1 seed | Q1 trial t, p% esetén: seed = `t * 111 + p` — mind a kiválasztást, mind a modell inicializációt befolyásolja |
| I8 | Q2 seed | Q2 trial t esetén: seed = `t * 222 + p` — csak a modell inicializációt befolyásolja (kiválasztás determinisztikus) |

---

## Frissítési napló

| Dátum | Változás |
|---|---|
| 2026-05-08 | Kezdeti fájl létrehozva az elemzés alapján |
| 2026-05-08 | Implementációs döntések hozzáadva; notebook javítva és futtatva |
