import random

# Base de datos local estática de sets de LEGO para demostración y consistencia
REAL_SETS = {
    "75078-1": {
        "name": 'Imperial Troop Transport (Star Wars Rebels)',
        "minifigures": [
            {
                        "ref": "sw0614",
                        "name": "Stormtrooper (Rebels) with Azure Vents",
                        "qty": 4
            }
],
        "parts": [
            {
                        "ref": "3004",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 8
            },
            {
                        "ref": "3001",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3020",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3022",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 12
            },
            {
                        "ref": "2877",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "59900",
                        "color_code": "36",
                        "color_hex": "#C91A09",
                        "color_name": "Trans-Red",
                        "qty": 4
            },
            {
                        "ref": "3003",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            }
]
    },
    "75280-1": {
        "name": '501st Legion Clone Troopers',
        "minifigures": [],
        "parts": [
            {
                        "ref": "75280stk01",
                        "color_code": "0",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 1
            },
            {
                        "ref": "87994",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "4735",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "37762",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "57899",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "58247",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 7
            },
            {
                        "ref": "64567",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "3023",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "15456",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "85861",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 3
            },
            {
                        "ref": "15403",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "54200",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "15068",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "32803",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "4599b",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "3705",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "63864",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "33909",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 3
            },
            {
                        "ref": "99780",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 9
            },
            {
                        "ref": "3004",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "4740",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 6
            },
            {
                        "ref": "3937",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "3023",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 4
            },
            {
                        "ref": "3021",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "3020",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "3034",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "15573",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 2
            },
            {
                        "ref": "2540",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 4
            },
            {
                        "ref": "87580",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 2
            },
            {
                        "ref": "99206",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "3176",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "85984",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 2
            },
            {
                        "ref": "3068",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 3
            },
            {
                        "ref": "26603",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "2412b",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 3
            },
            {
                        "ref": "2432",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "33909",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 3
            },
            {
                        "ref": "22385",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "47755",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 1
            },
            {
                        "ref": "44676",
                        "color_code": "272",
                        "color_hex": "#0D2654",
                        "color_name": "Dark Blue",
                        "qty": 4
            },
            {
                        "ref": "63965",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "11090",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "36840",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "44728",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "2877",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "44567b",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "44302",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "30304",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "30031",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "99774",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3710",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3021",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3795",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "61252",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "15573",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "48336",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "4032",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "15392",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "61409",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "85984pb127",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "42022",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3709",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "2412b",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "49668",
                        "color_code": "297",
                        "color_hex": "#899395",
                        "color_name": "Flat Silver",
                        "qty": 2
            },
            {
                        "ref": "4073",
                        "color_code": "297",
                        "color_hex": "#899395",
                        "color_name": "Flat Silver",
                        "qty": 4
            },
            {
                        "ref": "30374",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "4735",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "58176",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "28802",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "11215",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3004",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "4070",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "4216",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3941",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "43898",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "30387",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "6134",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "92582",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "23969",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3023",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 5
            },
            {
                        "ref": "3710",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3022",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3020",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3034",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "2445",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "4085d",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "15573",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 11
            },
            {
                        "ref": "4623b",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "11476",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "92280",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "14418",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "21445",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "4590",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "41740",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "4073",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "26047",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "15403",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "61409",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "54200",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "92946",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "24201",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "61678",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "85970",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3747b",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "4265c",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "32073",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "32064",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "41677",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "32001",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3069",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "2432",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "18674",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "66956",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "4073",
                        "color_code": "17",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 6
            }
]
    },
    "75218-1": {
        "name": 'X-Wing Starfighter',
        "minifigures": [],
        "parts": [
            {
                        "ref": "75218stk01",
                        "color_code": "0",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 1
            },
            {
                        "ref": "99781",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "3005",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "30552",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "4079",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "92738",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "3023",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 7
            },
            {
                        "ref": "3623",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 8
            },
            {
                        "ref": "3460",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "3022",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "2420",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "3020",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "85984",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "6553",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "32034",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 3
            },
            {
                        "ref": "32064",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "3701",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "62462",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "2780",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 8
            },
            {
                        "ref": "3709",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "32001",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "3023",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 4
            },
            {
                        "ref": "4274",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 2
            },
            {
                        "ref": "6558",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 4
            },
            {
                        "ref": "3024",
                        "color_code": "103",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 4
            },
            {
                        "ref": "3023",
                        "color_code": "103",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 4
            },
            {
                        "ref": "32952",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "87620",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "2489",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "4740",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "30553",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "2429c01",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3639",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "44300",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "30132",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3023",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 10
            },
            {
                        "ref": "3623",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3710",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 26
            },
            {
                        "ref": "3460",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3020",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 7
            },
            {
                        "ref": "15573",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "32028",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "11458",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 28
            },
            {
                        "ref": "4073",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 35
            },
            {
                        "ref": "15392",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "15301c01",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "61409",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "85984",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "85984pb127",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "15068",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "32209",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "10197",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "6538c",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "32064",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3894",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "32530",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "15712",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "43723",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "15573",
                        "color_code": "288",
                        "color_hex": "#184632",
                        "color_name": "Dark Green",
                        "qty": 2
            },
            {
                        "ref": "3005",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 1
            },
            {
                        "ref": "3010",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "3831",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "3830",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "3023",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 8
            },
            {
                        "ref": "3710",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "4477",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "85984",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 1
            },
            {
                        "ref": "3069",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 4
            },
            {
                        "ref": "3068",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 4
            },
            {
                        "ref": "32028",
                        "color_code": "28",
                        "color_hex": "#9F8F75",
                        "color_name": "Dark Tan",
                        "qty": 1
            },
            {
                        "ref": "3023",
                        "color_code": "39",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 2
            },
            {
                        "ref": "4032",
                        "color_code": "39",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 4
            },
            {
                        "ref": "18654",
                        "color_code": "297",
                        "color_hex": "#899395",
                        "color_name": "Flat Silver",
                        "qty": 8
            },
            {
                        "ref": "2412b",
                        "color_code": "297",
                        "color_hex": "#899395",
                        "color_name": "Flat Silver",
                        "qty": 6
            },
            {
                        "ref": "55982",
                        "color_code": "297",
                        "color_hex": "#899395",
                        "color_name": "Flat Silver",
                        "qty": 4
            },
            {
                        "ref": "3021",
                        "color_code": "6",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 1
            },
            {
                        "ref": "2714b",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "99781",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "18671",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3004",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 5
            },
            {
                        "ref": "3010",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3003",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "87087",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "22885",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "4589b",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "41531",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3937",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3938",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "60849",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "18738",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "99563",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3024",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3023",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "3460",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "2420",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 12
            },
            {
                        "ref": "3021",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3031",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "92280",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "18677",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "87580",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3176",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "11477",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 12
            },
            {
                        "ref": "2341",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "60219",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "4599b",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "4265c",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 12
            },
            {
                        "ref": "4519",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "32073",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "44294",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "57585",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "32039",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "6541",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "41677",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "11478",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "32054",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "62462",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3673",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3738",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3069",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "2431",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "43712",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "43719",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3795",
                        "color_code": "27",
                        "color_hex": "#BBE90B",
                        "color_name": "Lime",
                        "qty": 1
            },
            {
                        "ref": "34103",
                        "color_code": "42",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 2
            },
            {
                        "ref": "64567",
                        "color_code": "67",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 1
            },
            {
                        "ref": "96874",
                        "color_code": "25",
                        "color_hex": "#FE8A18",
                        "color_name": "Orange",
                        "qty": 1
            },
            {
                        "ref": "3003",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 1
            },
            {
                        "ref": "32952",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 2
            },
            {
                        "ref": "30414",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 2
            },
            {
                        "ref": "3062",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 1
            },
            {
                        "ref": "3710",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 2
            },
            {
                        "ref": "3020",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 3
            },
            {
                        "ref": "3795",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 3
            },
            {
                        "ref": "32062",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 4
            },
            {
                        "ref": "3707",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 1
            },
            {
                        "ref": "32530",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 1
            },
            {
                        "ref": "3005",
                        "color_code": "73",
                        "color_hex": "#5E748C",
                        "color_name": "Sand Blue",
                        "qty": 4
            },
            {
                        "ref": "3070",
                        "color_code": "73",
                        "color_hex": "#5E748C",
                        "color_name": "Sand Blue",
                        "qty": 11
            },
            {
                        "ref": "3069",
                        "color_code": "73",
                        "color_hex": "#5E748C",
                        "color_name": "Sand Blue",
                        "qty": 1
            },
            {
                        "ref": "11211",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 2
            },
            {
                        "ref": "3023",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 3
            },
            {
                        "ref": "3623",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 4
            },
            {
                        "ref": "4477",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 1
            },
            {
                        "ref": "3022",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 1
            },
            {
                        "ref": "2854",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 1
            },
            {
                        "ref": "3749",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 1
            },
            {
                        "ref": "6541",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 2
            },
            {
                        "ref": "32002",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 2
            },
            {
                        "ref": "21849pb04",
                        "color_code": "12",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 1
            },
            {
                        "ref": "4589b",
                        "color_code": "50",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 4
            },
            {
                        "ref": "30374",
                        "color_code": "33",
                        "color_hex": "#A5DBF5",
                        "color_name": "Trans-Light Blue",
                        "qty": 1
            },
            {
                        "ref": "4073",
                        "color_code": "17",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 4
            },
            {
                        "ref": "15303",
                        "color_code": "17",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 6
            },
            {
                        "ref": "63965",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "3005",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 3
            },
            {
                        "ref": "3622",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "4216",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "30414",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "4740",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "2429c01",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 3
            },
            {
                        "ref": "3023",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 8
            },
            {
                        "ref": "3623",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "3710",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 5
            },
            {
                        "ref": "3666",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 5
            },
            {
                        "ref": "3460",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "2420",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "3021",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 1
            },
            {
                        "ref": "3020",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 3
            },
            {
                        "ref": "3795",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "3832",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "2445",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "2639",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "3958",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "3839b",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 1
            },
            {
                        "ref": "92593",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 3
            },
            {
                        "ref": "85861",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "15403",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "x71",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "54200",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "85984",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 8
            },
            {
                        "ref": "4286",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "3297",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "3040",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "92946",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "28192",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 12
            },
            {
                        "ref": "15068",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "93273",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "85970",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "3676",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "4871",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "32000",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "6632",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "3070",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "3069",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 1
            },
            {
                        "ref": "63864",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "4162",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "3068",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 1
            },
            {
                        "ref": "14719",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "26603",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 6
            },
            {
                        "ref": "87079",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "6179",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "27263",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "43713",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "43723",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "43722",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 3
            },
            {
                        "ref": "14181",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "30355",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "30356",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "20309",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 12
            },
            {
                        "ref": "6070",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "15573",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 1
            },
            {
                        "ref": "4265c",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 3
            },
            {
                        "ref": "32064",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 2
            }
]
    },
    "75337-1": {
        "name": 'AT-TE Walker',
        "minifigures": [],
        "parts": [
            {
                        "ref": "75337stk01",
                        "color_code": "0",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 1
            },
            {
                        "ref": "4592c02",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "28802",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "11215",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 6
            },
            {
                        "ref": "3245c",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "2877",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 3
            },
            {
                        "ref": "4589b",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 10
            },
            {
                        "ref": "6154",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "30383",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 8
            },
            {
                        "ref": "44302",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 8
            },
            {
                        "ref": "60471",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "37762",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "57899",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "92738",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "58247",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "3023",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "3031",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "3958",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "3036",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "78257",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "2817",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "99206",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "64799",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "4073",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "49307",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "32802",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "4871",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "3713",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "32062",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "3705",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "32015",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "6536",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "6538c",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "26287",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 3
            },
            {
                        "ref": "6541",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "32064",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "3702",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "3743",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "32523",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "32524",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 1
            },
            {
                        "ref": "40490",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "41677",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 8
            },
            {
                        "ref": "32054",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 3
            },
            {
                        "ref": "32333",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "2780",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 37
            },
            {
                        "ref": "43093",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 6
            },
            {
                        "ref": "6558",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 16
            },
            {
                        "ref": "78258",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "62113",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "99780",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 5
            },
            {
                        "ref": "93274",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "3010",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3009",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3003",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3001",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "87087",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "2877",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "87081",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "61780",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "44359",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 12
            },
            {
                        "ref": "44301b",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "44302",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "3710",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 10
            },
            {
                        "ref": "3666",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "60479",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3022",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "2420",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3021",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3020",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 5
            },
            {
                        "ref": "3795",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3034",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3031",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3032",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "3036",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "15573",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 13
            },
            {
                        "ref": "32028",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "92107",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "4073",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 12
            },
            {
                        "ref": "4032",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 12
            },
            {
                        "ref": "60474",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "69755",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "61409",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "85984pb127",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3045",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3678b",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "6091",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "15068",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "93606",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "4287",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3660",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "32209",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 5
            },
            {
                        "ref": "32013",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "42003",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "6541",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "32000",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 10
            },
            {
                        "ref": "3701",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "32018",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3648",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "18654",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "41677",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "6629",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "15712",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "2412b",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 10
            },
            {
                        "ref": "6179",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "98138",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "27925",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 8
            },
            {
                        "ref": "47759",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "51739",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "4070",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "3023",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 6
            },
            {
                        "ref": "3710",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 7
            },
            {
                        "ref": "4477",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "3022",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 6
            },
            {
                        "ref": "3020",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "85984",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 1
            },
            {
                        "ref": "6538c",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 1
            },
            {
                        "ref": "3069",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "2431",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "6636",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "3068",
                        "color_code": "320",
                        "color_hex": "#720012",
                        "color_name": "Dark Red",
                        "qty": 2
            },
            {
                        "ref": "3010",
                        "color_code": "28",
                        "color_hex": "#9F8F75",
                        "color_name": "Dark Tan",
                        "qty": 3
            },
            {
                        "ref": "3899",
                        "color_code": "297",
                        "color_hex": "#899395",
                        "color_name": "Flat Silver",
                        "qty": 1
            },
            {
                        "ref": "4006",
                        "color_code": "297",
                        "color_hex": "#899395",
                        "color_name": "Flat Silver",
                        "qty": 1
            },
            {
                        "ref": "4073",
                        "color_code": "297",
                        "color_hex": "#899395",
                        "color_name": "Flat Silver",
                        "qty": 10
            },
            {
                        "ref": "2412b",
                        "color_code": "297",
                        "color_hex": "#899395",
                        "color_name": "Flat Silver",
                        "qty": 16
            },
            {
                        "ref": "98138pb008",
                        "color_code": "297",
                        "color_hex": "#899395",
                        "color_name": "Flat Silver",
                        "qty": 2
            },
            {
                        "ref": "87994",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "63965",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "36840",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "99781",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3005",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3004",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3001",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "11211",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "2653",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "44358",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "43898",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "3937",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "6134",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "53923",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "44567b",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "44302",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "92582",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "87544",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "91501",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "87421",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3023",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 12
            },
            {
                        "ref": "3623",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3710",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "4477",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "2420",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "3021",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3020",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3795",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "60470b",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "63868",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "10247",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "2476",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "4073",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 8
            },
            {
                        "ref": "2654",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "60474",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 11
            },
            {
                        "ref": "11213",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 10
            },
            {
                        "ref": "69754",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "54200",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 8
            },
            {
                        "ref": "85984",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "4286",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 8
            },
            {
                        "ref": "3297",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3040",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 12
            },
            {
                        "ref": "3049c",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "15571",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "92946",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "37352",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "11477",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "30165",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 3
            },
            {
                        "ref": "15068",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 29
            },
            {
                        "ref": "6081",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "85970",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "4287",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 8
            },
            {
                        "ref": "3747b",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "3665",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "4871",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "2449",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "4599b",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "4265c",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "32073",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "32184",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3700",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "3701",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "32531",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "73109",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "32316",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "41677",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "60484",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "4274",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 18
            },
            {
                        "ref": "3673",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3069",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "63864",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 8
            },
            {
                        "ref": "2431",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 6
            },
            {
                        "ref": "26603",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "87079",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "2432",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "33909",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "14769",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "27925",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 40
            },
            {
                        "ref": "61485",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "47755",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "48933",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "52031",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "50955",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "50956",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "26601",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "51739",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "43723",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "43722",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "2419",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "78443",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "78444",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 2
            },
            {
                        "ref": "54384",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "54383",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "6106",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "50305",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "50304",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 1
            },
            {
                        "ref": "96874",
                        "color_code": "25",
                        "color_hex": "#FE8A18",
                        "color_name": "Orange",
                        "qty": 1
            },
            {
                        "ref": "61190d",
                        "color_code": "25",
                        "color_hex": "#FE8A18",
                        "color_name": "Orange",
                        "qty": 4
            },
            {
                        "ref": "61190c",
                        "color_code": "25",
                        "color_hex": "#FE8A18",
                        "color_name": "Orange",
                        "qty": 3
            },
            {
                        "ref": "58247",
                        "color_code": "148",
                        "color_hex": "#575857",
                        "color_name": "Pearl Dark Gray",
                        "qty": 3
            },
            {
                        "ref": "3003",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 1
            },
            {
                        "ref": "44865",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 1
            },
            {
                        "ref": "3062",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 1
            },
            {
                        "ref": "3020",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 10
            },
            {
                        "ref": "3660",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 2
            },
            {
                        "ref": "32062",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 11
            },
            {
                        "ref": "3705",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 1
            },
            {
                        "ref": "3707",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 2
            },
            {
                        "ref": "62462",
                        "color_code": "70",
                        "color_hex": "#5C1E0F",
                        "color_name": "Reddish Brown",
                        "qty": 4
            },
            {
                        "ref": "4079",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 1
            },
            {
                        "ref": "78329",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 2
            },
            {
                        "ref": "3022",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 1
            },
            {
                        "ref": "78256",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 1
            },
            {
                        "ref": "4032",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 10
            },
            {
                        "ref": "49307",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 2
            },
            {
                        "ref": "3749",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 4
            },
            {
                        "ref": "3700",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 1
            },
            {
                        "ref": "33909",
                        "color_code": "19",
                        "color_hex": "#DFD1A5",
                        "color_name": "Tan",
                        "qty": 7
            },
            {
                        "ref": "4073",
                        "color_code": "108",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 4
            },
            {
                        "ref": "87544",
                        "color_code": "12",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 4
            },
            {
                        "ref": "4073",
                        "color_code": "12",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 4
            },
            {
                        "ref": "3023",
                        "color_code": "17",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 11
            },
            {
                        "ref": "98138",
                        "color_code": "17",
                        "color_hex": "#808080",
                        "color_name": "Unknown Color",
                        "qty": 6
            },
            {
                        "ref": "4073",
                        "color_code": "38",
                        "color_hex": "#F08F1C",
                        "color_name": "Trans-Orange",
                        "qty": 4
            },
            {
                        "ref": "99781",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 1
            },
            {
                        "ref": "30304",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 1
            },
            {
                        "ref": "4085d",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 2
            },
            {
                        "ref": "4265c",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 1
            },
            {
                        "ref": "4519",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 6
            }
]
    },
    "10692-1": {
        "name": "LEGO Classic Creative Bricks (50 Simple Parts Edition)",
        "minifigures": [],
        "parts": [
            {
                        "ref": "3001",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 4
            },
            {
                        "ref": "3002",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 4
            },
            {
                        "ref": "3003",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 4
            },
            {
                        "ref": "3004",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 6
            },
            {
                        "ref": "3005",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 10
            },
            {
                        "ref": "3010",
                        "color_code": "2",
                        "color_hex": "#00AA00",
                        "color_name": "Green",
                        "qty": 4
            },
            {
                        "ref": "3009",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 2
            },
            {
                        "ref": "3008",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 2
            },
            {
                        "ref": "3020",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 6
            },
            {
                        "ref": "3021",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 6
            },
            {
                        "ref": "3022",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 8
            },
            {
                        "ref": "3023",
                        "color_code": "2",
                        "color_hex": "#00AA00",
                        "color_name": "Green",
                        "qty": 12
            },
            {
                        "ref": "3024",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 10
            },
            {
                        "ref": "3034",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 2
            },
            {
                        "ref": "3035",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 2
            },
            {
                        "ref": "3031",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 4
            },
            {
                        "ref": "2420",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "3068",
                        "color_code": "2",
                        "color_hex": "#00AA00",
                        "color_name": "Green",
                        "qty": 4
            },
            {
                        "ref": "3069",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 6
            },
            {
                        "ref": "3070",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 8
            },
            {
                        "ref": "6636",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 4
            },
            {
                        "ref": "4162",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 2
            },
            {
                        "ref": "3038",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "3039",
                        "color_code": "2",
                        "color_hex": "#00AA00",
                        "color_name": "Green",
                        "qty": 6
            },
            {
                        "ref": "3040",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 6
            },
            {
                        "ref": "3298",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 4
            },
            {
                        "ref": "3037",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 4
            },
            {
                        "ref": "2412",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 8
            },
            {
                        "ref": "3710",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 10
            },
            {
                        "ref": "3622",
                        "color_code": "2",
                        "color_hex": "#00AA00",
                        "color_name": "Green",
                        "qty": 4
            },
            {
                        "ref": "3666",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 6
            },
            {
                        "ref": "3795",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 4
            },
            {
                        "ref": "4073",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 12
            },
            {
                        "ref": "3062",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 6
            },
            {
                        "ref": "22885",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 4
            },
            {
                        "ref": "32000",
                        "color_code": "2",
                        "color_hex": "#00AA00",
                        "color_name": "Green",
                        "qty": 4
            },
            {
                        "ref": "3700",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 4
            },
            {
                        "ref": "2877",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 6
            },
            {
                        "ref": "3659",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 2
            },
            {
                        "ref": "6141",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 12
            },
            {
                        "ref": "15573",
                        "color_code": "15",
                        "color_hex": "#FFFFFF",
                        "color_name": "White",
                        "qty": 8
            },
            {
                        "ref": "14719",
                        "color_code": "2",
                        "color_hex": "#00AA00",
                        "color_name": "Green",
                        "qty": 4
            },
            {
                        "ref": "18674",
                        "color_code": "4",
                        "color_hex": "#C91A09",
                        "color_name": "Red",
                        "qty": 4
            },
            {
                        "ref": "32013",
                        "color_code": "1",
                        "color_hex": "#0A3C9F",
                        "color_name": "Blue",
                        "qty": 4
            },
            {
                        "ref": "6536",
                        "color_code": "14",
                        "color_hex": "#F2CD37",
                        "color_name": "Yellow",
                        "qty": 4
            },
            {
                        "ref": "4274",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 10
            },
            {
                        "ref": "3673",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 10
            },
            {
                        "ref": "2780",
                        "color_code": "0",
                        "color_hex": "#1B1B1B",
                        "color_name": "Black",
                        "qty": 15
            },
            {
                        "ref": "3705",
                        "color_code": "84",
                        "color_hex": "#5A5A5A",
                        "color_name": "Dark Bluish Gray",
                        "qty": 4
            },
            {
                        "ref": "3003",
                        "color_code": "85",
                        "color_hex": "#A0A5A9",
                        "color_name": "Light Bluish Gray",
                        "qty": 4
            }
]
    }
}

# Paleta de colores comunes de LDraw para el generador dinámico
LDRAW_COLORS = [
    {"code": "85", "hex": "#A0A5A9", "name": "Light Bluish Gray"},
    {"code": "84", "hex": "#5A5A5A", "name": "Dark Bluish Gray"},
    {"code": "0", "hex": "#1B1B1B", "name": "Black"},
    {"code": "4", "hex": "#C91A09", "name": "Red"},
    {"code": "1", "hex": "#0A3C9F", "name": "Blue"},
    {"code": "14", "hex": "#F2CD37", "name": "Yellow"},
]

# Piezas LDraw comunes disponibles
LDRAW_PARTS = [
    {"ref": "3001", "name": "Brick 2x4"},
    {"ref": "3002", "name": "Brick 2x3"},
    {"ref": "3003", "name": "Brick 2x2"},
    {"ref": "3004", "name": "Brick 1x2"},
    {"ref": "3005", "name": "Brick 1x1"},
    {"ref": "3010", "name": "Brick 1x4"},
    {"ref": "3020", "name": "Plate 2x4"},
    {"ref": "3021", "name": "Plate 2x3"},
    {"ref": "3022", "name": "Plate 2x2"},
    {"ref": "3023", "name": "Plate 1x2"},
]

def get_set_data(set_id: str) -> dict:
    """
    Retorna el inventario de un set. Si el set existe en REAL_SETS, lo devuelve directamente.
    Si no, genera dinámicamente un set de LEGO realista para demostración de búsqueda bajo demanda.
    """
    # Limpiar formato de entrada (e.g. 75078 -> 75078-1)
    clean_id = set_id.strip()
    if "-" not in clean_id:
        clean_id = f"{clean_id}-1"
        
    if clean_id in REAL_SETS:
        return REAL_SETS[clean_id]
        
    # Inicializar generador determinista a partir del hash del Set ID para que devuelva
    # siempre el mismo inventario para un set concreto
    random.seed(hash(clean_id))
    
    num_parts = random.randint(5, 12)
    generated_parts = []
    
    # Seleccionar partes aleatorias del catálogo LDraw
    for _ in range(num_parts):
        part = random.choice(LDRAW_PARTS)
        color = random.choice(LDRAW_COLORS)
        qty = random.randint(2, 24)
        
        generated_parts.append({
            "ref": part["ref"],
            "color_code": color["code"],
            "color_hex": color["hex"],
            "color_name": color["name"],
            "qty": qty
        })
        
    # Minifiguras generadas
    minifigs = []
    num_minifigs = random.randint(0, 4)
    for i in range(num_minifigs):
        fig_id = f"fig-{random.randint(100, 999)}"
        minifigs.append({
            "ref": fig_id,
            "name": f"Minifigura Especial {fig_id.upper()}",
            "qty": random.randint(1, 2)
        })
        
    return {
        "name": f"Set Genérico Lego #{clean_id}",
        "minifigures": minifigs,
        "parts": generated_parts
    }
