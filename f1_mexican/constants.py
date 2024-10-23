import numpy as np

tyre_compounds = {
    "HARD": "C1 (Hard): Toughest, durable, slowest option",
    "MEDIUM": "C2 (Medium): Balanced, versatile, mid-speed",
    "SOFT": "C3 (Soft): Grip-focused, fast, moderate durability",
    "HYPERSOFT": "C4 (Super Soft): High-grip, rapid wear",
    "ULTRASOFT": "C5 (Ultra Soft): Maximum grip, least durability",
    "INTERMEDIATE": "Intermediate: Light rain, wet tracks",
    "WET": "Full Wet: Extreme rain, maximum evacuation",
}

tyre_compounds_embeddings = {
    "HARD": [0.9592574834823608, 0.2825334072113037],
    "MEDIUM": [0.9999638795852661, 0.008497972041368484],
    "SOFT": [0.982682466506958, 0.1852976679801941],
    "SUPERSOFT": [0.9986254572868347, -0.05241385102272034],
    "HYPERSOFT": [-0.8185507655143738, 0.5744342803955078],
    "ULTRASOFT": [0.9489226341247559, 0.31550878286361694],
    "INTERMEDIATE": [0.8133333325386047, 0.5817979574203491],
    "WET": [0.5332446694374084, 0.8459610342979431],
    "UNKNOWN": [np.nan, np.nan],
    "nan": [np.nan, np.nan],
}

f1_drivers = {
    "GAS": "Pierre Gasly",
    "PER": "Sergio Perez",
    "ALO": "Fernando Alonso",
    "LEC": "Charles Leclerc",
    "STR": "Lance Stroll",
    "VAN": "Stoffel Vandoorne",
    "MAG": "Kevin Magnussen",
    "HUL": "Nico Hulkenberg",
    "HAR": "Nyck de Vries",
    "RIC": "Daniel Ricciardo",
    "OCO": "Esteban Ocon",
    "VER": "Max Verstappen",
    "SIR": "George Russell",
    "HAM": "Lewis Hamilton",
    "VET": "Sebastian Vettel",
    "SAI": "Carlos Sainz",
    "RAI": "Kimi Räikkönen",
    "BOT": "Valtteri Bottas",
    "GRO": "Romain Grosjean",
    "ERI": "Marcus Ericsson",
    "ALB": "Alexander Albon",
    "KVY": "Daniil Kvyat",
    "GIO": "Antonio Giovinazzi",
    "RUS": "George Russell",
    "KUB": "Robert Kubica",
    "NOR": "Lando Norris",
    "LAT": "Nicholas Latifi",
    "MAZ": "Nikita Mazepin",
    "MSC": "Mick Schumacher",
    "TSU": "Yuki Tsunoda",
    "ZHO": "Zhou Guanyu",
    "SAR": "Logan Sargeant",
    "PIA": "Oscar Piastri",
    "BEA": "Olivier Bearman",
}

f1_drivers_embeddings = {
    "GAS": [-0.9983760714530945, -0.05696616694331169],
    "PER": [-0.5922185182571411, -0.8057774305343628],
    "ALO": [0.042458515614271164, -0.9990982413291931],
    "LEC": [0.9664713144302368, 0.2567746639251709],
    "STR": [0.995934784412384, 0.09007684141397476],
    "VAN": [0.514300525188446, -0.8576100468635559],
    "MAG": [-0.9876480102539062, -0.15668871998786926],
    "HUL": [-0.8260842561721802, -0.5635467171669006],
    "HAR": [-0.7487896084785461, -0.6628077626228333],
    "RIC": [-0.5201737284660339, -0.8540604710578918],
    "OCO": [0.45937782526016235, -0.8882409930229187],
    "VER": [-0.9986558556556702, -0.05183197930455208],
    "SIR": [-0.9988524913787842, 0.04789366573095322],
    "HAM": [-0.8970533609390259, -0.44192227721214294],
    "VET": [-0.7715311646461487, -0.6361915469169617],
    "SAI": [0.39811190962791443, -0.9173368811607361],
    "RAI": [0.3659997880458832, -0.9306149482727051],
    "BOT": [-0.7284786701202393, -0.6850684285163879],
    "GRO": [-0.7495142817497253, 0.6619881391525269],
    "ERI": [-0.7570468783378601, -0.6533604264259338],
    "ALB": [-0.3867266774177551, -0.9221943616867065],
    "KVY": [-0.9413043260574341, 0.3375591039657593],
    "GIO": [-0.8026657104492188, -0.596429169178009],
    "RUS": [-0.9988524913787842, 0.04789366573095322],
    "KUB": [-0.8654263019561768, 0.5010362863540649],
    "NOR": [0.7976977825164795, -0.6030573844909668],
    "LAT": [-0.8527942895889282, -0.5222468972206116],
    "MAZ": [-0.9992873072624207, -0.03774714842438698],
    "MSC": [0.9985706806182861, 0.05344634875655174],
    "TSU": [-0.533794105052948, -0.8456144332885742],
    "ZHO": [-0.46488508582115173, 0.8853710293769836],
    "SAR": [0.5869675278663635, 0.8096104264259338],
    "PIA": [0.17536179721355438, 0.9845040440559387],
    "BEA": [0.4983256757259369, -0.8669899106025696]
}

f1_teams = [
    "Toro Rosso",
    "Racing Point",
    "McLaren",
    "Sauber",
    "Williams",
    "Haas F1 Team",
    "Renault",
    "Red Bull Racing",
    "Mercedes",
    "Ferrari",
    "Alfa Romeo Racing",
    "AlphaTauri",
    "Aston Martin",
    "Alpine",
    "Alfa Romeo",
    "RB"
]

f1_teams_embeddings = {
    "Toro Rosso": [-0.7968116402626038, -0.6042277812957764],
    "Racing Point": [-0.39232078194618225, 0.919828474521637],
    "McLaren": [-0.7953025698661804, -0.6062126755714417],
    "Kick Sauber": [0.4442012310028076, -0.8959270119667053],
    "Williams": [0.2608851492404938, 0.9653698801994324],
    "Haas F1 Team": [-0.9946078062057495, 0.1037079393863678],
    "Renault": [-0.9239233136177063, -0.38257765769958496],
    "Red Bull Racing": [-0.7991177439689636, -0.601174533367157],
    "Mercedes": [-0.027565820142626762, -0.9996199607849121],
    "Ferrari": [-0.609474241733551, -0.7928059101104736],
    "Alfa Romeo Racing": [-0.9727749228477478, -0.23175176978111267],
    "AlphaTauri": [-0.6645904183387756, -0.747207760810852],
    "Aston Martin": [-0.6835925579071045, -0.7298638224601746],
    "Alpine": [-0.10708858072757721, -0.9942494630813599],
    "Alfa Romeo": [-0.7892884612083435, -0.6140226125717163],
    "RB": [-0.9996195435523987, 0.027582256123423576],
}


grid_pos = {
    "FRONT": "Front-Runners: Conservative strategy, defensive pitting to maintain lead.",
    "MID": "Mid-Pack: Aggressive two-stop strategy, tire advantage for overtakes.",
    "BACK": "Back-Markers: Unconventional strategies, gamble on weather and safety cars.",
}

grid_pos_embeddings = {
    "FRONT": [0.9162107110023499, 0.400696724653244],
    "MID": [0.9990228414535522, 0.044197872281074524],
    "BACK": [0.9205362200737, 0.39065730571746826],
}

unique_grand_prix_names = {
    "Australian Grand Prix": ["Australian Grand Prix"],  # Albert Park Circuit
    "Bahrain Grand Prix": [
        "Bahrain Grand Prix",
        "Sakhir Grand Prix",
    ],  # Same layout (Bahrain International Circuit)
    "Chinese Grand Prix": ["Chinese Grand Prix"],  # Shanghai International Circuit
    "Azerbaijan Grand Prix": ["Azerbaijan Grand Prix"],  # Baku City Circuit
    "Spanish Grand Prix": ["Spanish Grand Prix"],  # Circuit de Barcelona-Catalunya
    "Monaco Grand Prix": ["Monaco Grand Prix"],  # Circuit de Monaco
    "Canadian Grand Prix": ["Canadian Grand Prix"],  # Circuit Gilles Villeneuve
    "French Grand Prix": ["French Grand Prix"],  # Circuit Paul Ricard
    "Austrian Grand Prix": [
        "Austrian Grand Prix",
        "Styrian Grand Prix",
    ],  # Same layout (Red Bull Ring)
    "British Grand Prix": ["British Grand Prix"],  # Silverstone Circuit
    "German Grand Prix": ["German Grand Prix"],  # Hockenheimring
    "Eifel Grand Prix": ["Eifel Grand Prix"],  # Nürburgring Grand Prix Circuit
    "Hungarian Grand Prix": ["Hungarian Grand Prix"],  # Hungaroring
    "Belgian Grand Prix": ["Belgian Grand Prix"],  # Circuit de Spa-Francorchamps
    "Italian Grand Prix": ["Italian Grand Prix"],  # Autodromo Nazionale Monza
    "Singapore Grand Prix": ["Singapore Grand Prix"],  # Marina Bay Street Circuit
    "Russian Grand Prix": ["Russian Grand Prix"],  # Sochi Autodrom
    "Japanese Grand Prix": ["Japanese Grand Prix"],  # Suzuka Circuit
    "United States Grand Prix": [
        "United States Grand Prix"
    ],  # Circuit of the Americas (COTA)
    "Mexican Grand Prix": [
        "Mexican Grand Prix",
        "Mexico City Grand Prix",
    ],  # Same layout (Autódromo Hermanos Rodríguez)
    "Brazilian Grand Prix": [
        "Brazilian Grand Prix",
        "São Paulo Grand Prix",
    ],  # Same layout (Autódromo José Carlos Pace - Interlagos)
    "Abu Dhabi Grand Prix": ["Abu Dhabi Grand Prix"],  # Yas Marina Circuit
    "Tuscan Grand Prix": ["Tuscan Grand Prix"],  # Mugello Circuit
    "Portuguese Grand Prix": [
        "Portuguese Grand Prix"
    ],  # Autódromo Internacional do Algarve (Portimão)
    "Emilia Romagna Grand Prix": [
        "Emilia Romagna Grand Prix"
    ],  # Autodromo Enzo e Dino Ferrari (Imola)
    "Turkish Grand Prix": ["Turkish Grand Prix"],  # Istanbul Park
    "Dutch Grand Prix": ["Dutch Grand Prix"],  # Circuit Zandvoort
    "Qatar Grand Prix": ["Qatar Grand Prix"],  # Losail International Circuit
    "Saudi Arabian Grand Prix": ["Saudi Arabian Grand Prix"],  # Jeddah Street Circuit
    "Miami Grand Prix": ["Miami Grand Prix"],  # Miami International Autodrome
    "Las Vegas Grand Prix": ["Las Vegas Grand Prix"],  # Las Vegas Street Circuit
}

grand_prix_embeddings = {
    "Australian Grand Prix": [-0.8803253769874573, 0.47437041997909546],
    "Bahrain Grand Prix": [-0.7679700255393982, 0.6404857635498047],
    "Chinese Grand Prix": [-0.3710212707519531, -0.9286243319511414],
    "Azerbaijan Grand Prix": [-0.7120967507362366, -0.7020813226699829],
    "Spanish Grand Prix": [-0.6628386974334717, -0.7487622499465942],
    "Monaco Grand Prix": [-0.9855400919914246, -0.16944225132465363],
    "Canadian Grand Prix": [-0.3512243628501892, 0.9362913966178894],
    "French Grand Prix": [-0.9915491938591003, 0.12973089516162872],
    "Austrian Grand Prix": [-0.9997895956039429, 0.020514944568276405],
    "British Grand Prix": [-0.28731653094291687, 0.9578356742858887],
    "German Grand Prix": [-0.7895243167877197, 0.6137192845344543],
    "Eifel Grand Prix": [-0.9810302257537842, -0.1938548982143402],
    "Hungarian Grand Prix": [-0.9990700483322144, -0.04311743006110191],
    "Belgian Grand Prix": [-0.9801609516143799, 0.1982034295797348],
    "Italian Grand Prix": [-0.951958954334259, -0.30622562766075134],
    "Singapore Grand Prix": [-0.8431386947631836, -0.5376960635185242],
    "Russian Grand Prix": [-0.9576643705368042, -0.2878868281841278],
    "Japanese Grand Prix": [-0.9083727598190308, 0.41816121339797974],
    "United States Grand Prix": [-0.9276883602142334, 0.3733553886413574],
    "Mexican Grand Prix": [-0.7735356688499451, 0.6337527632713318],
    "Brazilian Grand Prix": [-0.7322566509246826, 0.6810287833213806],
    "Abu Dhabi Grand Prix": [-0.8930231332778931, 0.45001089572906494],
    "Tuscan Grand Prix": [-0.9360460042953491, -0.3518775999546051],
    "Portuguese Grand Prix": [-0.9846985936164856, 0.17426618933677673],
    "Emilia Romagna Grand Prix": [-0.9200829267501831, 0.3917236030101776],
    "Turkish Grand Prix": [-0.878420352935791, -0.477888822555542],
    "Dutch Grand Prix": [-0.9996195435523987, 0.027582256123423576],
    "Qatar Grand Prix": [0.46147799491882324, 0.8871516585350037],
    "Saudi Arabian Grand Prix": [-0.8728778958320618, -0.4879387319087982],
    "Miami Grand Prix": [-0.7482759952545166, 0.6633875370025635],
    "Las Vegas Grand Prix": [-0.8561841249465942, 0.516670823097229],
}

team_mapping = {
    "AlphaTauri": ["Toro Rosso", "AlphaTauri"],
    "Aston Martin": ["Force India", "Racing Point", "Aston Martin"],
    "McLaren": ["McLaren"],
    "Williams": ["Williams"],
    "Haas F1 Team": ["Haas F1 Team"],
    "Alpine": ["Renault", "Alpine"],
    "Red Bull Racing": ["Red Bull Racing"],
    "RB": ["RB"],  # Separate entity
    "Mercedes": ["Mercedes"],
    "Kick Sauber": ["Sauber", "Alfa Romeo", "Alfa Romeo Racing", "Kick Sauber"],
    "Ferrari": ["Ferrari"],
}

drivers_teams = {
    "NOR": "McLaren",
    "PIA": "McLaren",
    "VER": "Red Bull Racing",
    "PER": "Red Bull Racing",
    "LEC": "Ferrari",
    "SAI": "Ferrari",
    "HAM": "Mercedes",
    "RUS": "Mercedes",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "TSU": "RB",
    "LAW": "RB",
    "RIC": "RB",
    "HUL": "Haas F1 Team",
    "MAG": "Haas F1 Team",
    "BEA": "Haas F1 Team",
    "ALB": "Williams",
    "COL": "Williams",
    "SAR": "Williams",
    "GAS": "Alpine",
    "OCO": "Alpine",
    "BOT": "Kick Sauber",
    "ZHO": "Kick Sauber"
}


