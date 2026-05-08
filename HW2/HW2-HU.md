# Házi feladat: Célzott és nem célzott poisoning támadás

A házi feladat két részből áll. Először az első házi feladatban megvalósított IDS detektor bináris változata ellen kell nem célzott adatszennyezést végrehajtani (untargeted data poisoning), majd a második részben célzott adatszennyezést (targeted poisoning). 

# Támadó modell

A támadó hozzáfér a teszt és tanítóadathoz is. A támadás során képes fabrikált mintákat hozzáadni a tanítóadathoz, de a már benne levőket nem tudja módosítani. Továbbá a támadó ismeri a detektor hyper-paramétereit (pl. a háló architektúráját, optimalizációs algoritmus, stb.) de a megtámadott modell paramétereit nem. 

# Előkészítés

Az első házi feladathoz hasonlóan tanítson be egy IDS detektort annyi különbséggel, hogy a tanító és teszt adat legyen binárisan címkézett (minden támadás legyen az 1-es osztályban, míg minden "normal" címkéjű legyen a 0-ás osztályban). Mellékelve találja az IDS modell architektúráját, ami megegyezik az első házi feladat detektorával annyi különbséggel, hogy a modell kimenete vagy egyetlen logit érték ha a loss függvény BCEWithLogitsLoss, vagy egy sigmoid érték ha a loss függvény BCELoss. Minden más hyperparaméter (optimizer, learning rate, batch méret, stb.) megegyezik az első házi feladatban használtakkal. 

A tanítóadatot vágja fel véletlen módon validációs és tanítóadatra 20-80% arányban (80% tanító, 20% validációs adat). Mostantól tanítóadat alatt ezt 80%-os részt értjük, validációs pontosság alatt pedig a validációs részen számolt pontosságot!

Továbbá mindkét feladathoz szükséges öt különböző surrogate modell betanítása a **tesztadaton** az első házi feladatban leírt módon (mindegyik teljes újratanítás véletlen inicializált model paraméterekkel és véletlen mini-batch sorrenddel). A támadó hozzáfér ezekhez a betanított surrogate modellekhez és felhasználhatja a támadásokhoz.

# 1. feladat: Untargeted poisoning

A támadó célja, hogy az IDS detektor pontosságát lerontsa a validációs adaton: veszi a tanítási adat $p$ %-át, ezek címkéit invertálja (támadásról nem támadásra, nem támadásról támadásra), majd az átcímkézett adatot hozzáadja a tanítási adathoz. Az így kibővített adaton kell újratanítani a modellt, és meghatározni a kapott modell validációs pontosságát.

## Kérdések

1. A támadó véletlenül választja az átcímkézendő mintákat az összes tanítóadat közül. Mi lesz a beszennyezett adaton tanított modell validációs pontossága ha $p$ értéke 30, 50 és 70%? Minden esetben értékelje ki a modell átlagos validációs pontosságát legalább 5 különböző tanításon, és ábrázolja box-ploton az eredményeket! 
2. A támadó a legnagyobb loss értékű mintákat választja átcímkézésre az összes tanítóadat közül. Ehhez minden tanító minta loss értékét kiszámolja a minta invertált címkéjére nézve az előkészítésben kiszámolt 5 surrogate modell mindegyikén. Minden mintára az 5 darab loss értéket átlagolja, majd kiválasztja a legnagyobb átlagos loss értékű minták felső $p$ %-át, és hozzáadja ezeket invertált címkével a tanítóadathoz. Mi lesz a beszennyezett adaton betanított modell validációs pontossága ha $p$ értéke 30, 50 és 70%? Minden esetben értékelje ki a modell átlagos validációs pontosságát legalább 5 különböző tanításon, és ábrázolja box-ploton az eredményeket!
3. Hasonlítsa össze a fenti két esetben kapott eredményeket! Melyik hatékonyabb és miért?
4. Elemezze a támadások megvalósíthatóságát a gyakorlatban! Javasoljon bármilyen védekezést amivel a támadás sikeressége csökkenthető (nem kell implementálni)!

# 2. feladat: Targeted posioning

A feladat egy célzott adatszennyezéses támadás megvalósítása. A 10 darab félreklasszifikálandó (cél) mintákat válassza ki úgy, hogy azok a 
- teszt (és nem validációs!) adatból származzanak,
- eredetileg mindegyik támadásként van címkézve,
- a clean IDS detektor is támadásként klasszifikálja őket,
- a loss értékük a nem támadott címkére nézve minimális, vagy is olyan támadások amiben a clean IDS detektor eleve bizonytalan.

 A cél, hogy az így kiválasztott összesen 10 darab target mintát egy szennyezett adaton betanított modell félreklasszifikálja deployment után a tesztfázisban. A támadó ehhez kiválaszt $p$ darab base mintát a teszthalmazból, és módosítja a mintát úgy, hogy az így kapott poison mintán számolt gradiens illeszkedni fog az invertált címkével ellátott target mintán számolt gradienssel. Vagyis a hozzáadott poison minták "szimulálják" az "nem támadás" osztályra átcímkézett target minta gradiensét a tanítás során, mintha $x_t$ ezzel a címkével benne lenne a tanítóadatban. A WiB támadás során az alábbi alignment loss értéket kell minimalizálni $\delta_i$ szerint ($1\leq j \leq p$) adott $x_t$ target mintára:
$$
\min_{\{\delta_1, \ldots, \delta_p\}} \frac{1}{5}\sum_{j=1}^5 \sum_{i=1}^p (1 - \cos\angle(\nabla_\theta loss_f(\theta_j, x_t, y_t^{adv}), \nabla_\theta loss_f(\theta_j, x_i + \delta_i, y_i)))
$$
ahol $y_t^{adv}=\overline{y_t}=y_i$ a támadó által választott célosztály, $\{\theta_1, \ldots, \theta_5\}$ pedig az előkészítés során betanított 5 darab surrogate modell amivel a támadó rendelkezik. A base mintákat úgy választja, hogy azok gradiense és a target minta gradiense eleve a leginkább hasonlóak legyenek az összes $y_t^{adv}$ címkéjű teszt minta közül: Minden $y_t^{adv}$ címkéjű tesztminta és adott  $(x_t, y_t^{adv})$ target minta gradiensének a koszinusz hasonlóságát kiszámolja a $\{\theta_1, \ldots, \theta_5\}$ modellek mindegyikén, ezt az 5 darab hasonlósági értéket átlagolja minden tesztminta esetén, majd kiválasztja a $p$ legnagyobb átlagos koszinusz hasonlósággal rendelkező tesztmintákat. A támadás kimenete a $\{(x_i+\delta_i, y_i)\}_{i=1}^p$ poison minták halmaza, amit a tanítóadathoz kell adni, és a modellt az így szennyezett adaton újratanítani. 

További részletekért lásd a Witches' Brew (WiB) támadást az előadás anyagában. (Megjegyzés: Az előadásban a  $\frac{1}{5}\sum_{j=1}^5 (1 - cos\angle(\nabla_\theta loss_f(\theta_j, x_t, y_t^{adv}),  \sum_{i=1}^p \nabla_\theta loss_f(\theta_j, x_i + \delta_i, y_i)))$ formula szerepelt, amit ugyan gyorsabb kiszámolni de itt pontatlanabb támadást eredményez) 

# Kérdések:
1. Hajtsa végre a fenti támadást minden egyes target mintára ha $p\in\{5, 15\}$ és a módosított feature értékek nem lehetnek nagyobbak/kisebbek mint a tanítóadat maximális/minimális skálázott feature értékei (más plauzibilitási ellenőrzés nem szükséges)! Számolja ki a támadás átlagos pontosságát minden egyes $p$ értékre, vagyis a sikeresen félreklasszifikált target minták arányát! A támadás akkor sikeres, ha a szennyezett tanítóadaton újratanított modell (random inicializálva véletlen mini-batch választás esetén) az $x_t$ bemenetet nem támadott osztályba ($y_t^{adv}$) sorolja.
2. Elemezze a támadás megvalósíthatóságát a gyakorlatban! Javasoljon bármilyen védekezést amivel a támadás sikeressége csökkenthető (nem kell implementálni)!

*Megjegyzések:* 
- A base mintákból számolt input vektor minden koordinátáját módosíthatja a támadó  (de az eredeti $y_i$ címkét nem)!
- A $loss_{align}$ minimalizálását érdemes PGD-vel végezni (pl. SGD optimalizációval), dinamikusan csökkenő learning rate értékkel. Pl: `ReduceLROnPlateau(optimizer, 'min', patience=75, eps=1e-06)` használata estén nem szükséges 1000-nél több PGD iteráció. Ha az 1000. iteráció után kapott poison minták sem eredményeznek félreklasszifikálást, akkor a támadás tekinthető sikertelennek.
	
# Beadás módja

Egyetlen ZIP-fájlt kell létrehozni és a Moodle-be feltölteni, amelynek neve: `mlsec_hw_2_<NEPTUN_ID1>-<NEPTUN_ID2>.zip` (cserélje le `<NEPTUN_ID>`-t a saját és csapattagja neptun azonosítójára), amely tartalmazza a megoldást (Python Notebook) és ugyanebben a notebook-ban a kérdésekre adott válaszokat. 
A forráskódot legyen kommentezve, adatot (binárist, numpy tömböket stb.) nem kell feltölteni! Csapatonként elég egy beadás.