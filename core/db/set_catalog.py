import random

# Base de datos local estática de sets de LEGO para demostración y consistencia
REAL_SETS = {
    "75078-1": {
        "name": "Imperial Troop Transport",
        "minifigures": [
            {
                "ref": "sw0578",
                "name": "Imperial Stormtrooper - Printed Legs, Dark Azure Helmet Vents",
                "qty": 2
            },
            {
                "ref": "sw0617",
                "name": "Imperial Stormtrooper - Printed Legs, Dark Azure Helmet Vents, Frown",
                "qty": 2
            }
        ],
        "parts": [
            {
                "ref": "2877",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Brick, Modified 1 x 2 with Grille / Fluted Profile"
            },
            {
                "ref": "30414",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Brick, Modified 1 x 4 with Studs on Side"
            },
            {
                "ref": "15391",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Minifigure, Weapon Gun, Mini Blaster / Shooter"
            },
            {
                "ref": "32054",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Pin 3L with Friction Ridges and Stop Bush"
            },
            {
                "ref": "61780",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Container, Box 2 x 2 x 2 - Top Opening"
            },
            {
                "ref": "3022",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Plate 2 x 2"
            },
            {
                "ref": "3795",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Plate 2 x 6"
            },
            {
                "ref": "3832",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Plate 2 x 10"
            },
            {
                "ref": "2654",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Plate, Round 2 x 2 with Rounded Bottom (Boat Stud)"
            },
            {
                "ref": "15392",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Projectile Launcher Part, Trigger for Gun, Mini Blaster / Shooter"
            },
            {
                "ref": "2412b",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 7,
                "name": "Tile, Modified 1 x 2 Grille with Bottom Groove"
            },
            {
                "ref": "3004",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Brick 1 x 2"
            },
            {
                "ref": "2653",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Brick, Modified 1 x 4 with Channel"
            },
            {
                "ref": "87620",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Brick, Modified Facet 2 x 2"
            },
            {
                "ref": "2335",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Flag 2 x 2 Square with Flat Edge"
            },
            {
                "ref": "87552",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Panel 1 x 2 x 2 with Side Supports - Hollow Studs"
            },
            {
                "ref": "3710",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Plate 1 x 4"
            },
            {
                "ref": "3795",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Plate 2 x 6"
            },
            {
                "ref": "2445",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Plate 2 x 12"
            },
            {
                "ref": "3839b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Plate, Modified 1 x 2 with Bar Handles - Flat Ends, Low Attachment"
            },
            {
                "ref": "85984",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Slope 30 1 x 2 x 2/3"
            },
            {
                "ref": "3040",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Slope 45 2 x 1"
            },
            {
                "ref": "60481",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Slope 65 2 x 1 x 2"
            },
            {
                "ref": "2449",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Slope, Inverted 75 2 x 1 x 3"
            },
            {
                "ref": "6541",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Brick 1 x 1 with Hole"
            },
            {
                "ref": "32000",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Brick 1 x 2 with Holes"
            },
            {
                "ref": "61184",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Pin 1/2 with 2L Bar Extension (Flick Missile)"
            },
            {
                "ref": "3068",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Tile 2 x 2"
            },
            {
                "ref": "14769",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Tile, Round 2 x 2 with Bottom Stud Holder"
            },
            {
                "ref": "3679",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Turntable 2 x 2 Plate, Top"
            },
            {
                "ref": "51739",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Wedge, Plate 2 x 4"
            },
            {
                "ref": "2419",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Wedge, Plate 3 x 6 Cut Corners"
            },
            {
                "ref": "3024",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Plate 1 x 1"
            },
            {
                "ref": "3023",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 6,
                "name": "Plate 1 x 2"
            },
            {
                "ref": "3023",
                "color_code": "13",
                "color_hex": "#625E51",
                "color_name": "Trans-Brown",
                "qty": 6,
                "name": "Plate 1 x 2"
            },
            {
                "ref": "4589b",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 2,
                "name": "Cone 1 x 1 with Top Groove"
            },
            {
                "ref": "4073",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 12,
                "name": "Plate, Round 1 x 1"
            },
            {
                "ref": "3022",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2,
                "name": "Plate 2 x 2"
            },
            {
                "ref": "3020",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Plate 2 x 4"
            },
            {
                "ref": "2540",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4,
                "name": "Plate, Modified 1 x 2 with Bar Handle on Side - Free Ends"
            },
            {
                "ref": "4073",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2,
                "name": "Plate, Round 1 x 1"
            },
            {
                "ref": "3680",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Turntable 2 x 2 Plate, Base"
            }
        ]
    },
    "911943-1": {
        "name": "Luke Skywalker foil pack #1 (Star Wars)",
        "minifigures": [
            {
                "ref": "sw0778",
                "name": "Luke Skywalker (Tatooine, White Legs, Stern / Smile Face Print)",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "64567",
                "color_code": "297",
                "color_hex": "#899395",
                "color_name": "Flat Silver",
                "qty": 1,
                "name": "Minifigure, Weapon Lightsaber Hilt Straight"
            },
            {
                "ref": "30374",
                "color_code": "15",
                "color_hex": "#AEE9EF",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Bar 4L (Lightsaber Blade / Wand)"
            }
        ]
    },
    "75280-1": {
        "name": "501st Legion Clone Troopers",
        "minifigures": [
            {
                "ref": "sw1093",
                "name": "Clone Trooper, 501st Legion",
                "qty": 3
            },
            {
                "ref": "sw1094",
                "name": "Clone Trooper Captain, 501st Legion",
                "qty": 1
            },
            {
                "ref": "sw1095",
                "name": "Battle Droid",
                "qty": 2
            }
        ],
        "parts": [
            {
                "ref": "75280stk01",
                "color_code": "0",
                "color_hex": "#808080",
                "color_name": "Various",
                "qty": 1
            },
            {
                "ref": "87994",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "4735",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "37762",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "57899",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "58247",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 7
            },
            {
                "ref": "64567",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "3023",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "15456",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "85861",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 3
            },
            {
                "ref": "15403",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "54200",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "15068",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "32803",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "4599b",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "3705",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "63864",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "33909",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 3
            },
            {
                "ref": "99780",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 9
            },
            {
                "ref": "3004",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1
            },
            {
                "ref": "4740",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 6
            },
            {
                "ref": "3937",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1
            },
            {
                "ref": "3023",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 4
            },
            {
                "ref": "3021",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1
            },
            {
                "ref": "3020",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1
            },
            {
                "ref": "3034",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1
            },
            {
                "ref": "15573",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 2
            },
            {
                "ref": "2540",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 4
            },
            {
                "ref": "87580",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 2
            },
            {
                "ref": "99206",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1
            },
            {
                "ref": "3176",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1
            },
            {
                "ref": "85984",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 2
            },
            {
                "ref": "3068",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 3
            },
            {
                "ref": "26603",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1
            },
            {
                "ref": "2412b",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 3
            },
            {
                "ref": "2432",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1
            },
            {
                "ref": "33909",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 3
            },
            {
                "ref": "22385",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1
            },
            {
                "ref": "47755",
                "color_code": "7",
                "color_hex": "#0032B1",
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
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "11090",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "36840",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "44728",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "2877",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "44567b",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "44302",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "30304",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "30031",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "99774",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3710",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3021",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3795",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "61252",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "15573",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "48336",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "4032",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 3
            },
            {
                "ref": "15392",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 3
            },
            {
                "ref": "61409",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "85984pb127",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "42022",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3709",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "2412b",
                "color_code": "85",
                "color_hex": "#6B6D67",
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
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 3
            },
            {
                "ref": "4735",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "58176",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "28802",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6
            },
            {
                "ref": "11215",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3004",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 3
            },
            {
                "ref": "4070",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "4216",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3941",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "43898",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "30387",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "6134",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "92582",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "23969",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3023",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 5
            },
            {
                "ref": "3710",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3022",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3020",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3034",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "2445",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "4085d",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "15573",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 11
            },
            {
                "ref": "4623b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "11476",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 3
            },
            {
                "ref": "92280",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "14418",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "21445",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "4590",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "41740",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "4073",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "26047",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "15403",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "61409",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "54200",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "92946",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "24201",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "61678",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "85970",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3747b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "4265c",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "32073",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "32064",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "41677",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "32001",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3069",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "2432",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "18674",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "66956",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "4073",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 6
            }
        ]
    },
    "75218-1": {
        "name": "X-Wing Starfighter",
        "minifigures": [
            {
                "ref": "sw0886",
                "name": "Luke Skywalker (X-Wing Pilot)",
                "qty": 1
            },
            {
                "ref": "sw0536",
                "name": "R2-D2",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "75218stk01",
                "color_code": "0",
                "color_hex": "#808080",
                "color_name": "Various",
                "qty": 1
            },
            {
                "ref": "99781",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "3005",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "30552",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "4079",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "92738",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "3023",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 7
            },
            {
                "ref": "3623",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 8
            },
            {
                "ref": "3460",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "3022",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "2420",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "3020",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "85984",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "6553",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "32034",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 3
            },
            {
                "ref": "32064",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "3701",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "62462",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 8
            },
            {
                "ref": "3709",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "32001",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "3023",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 4
            },
            {
                "ref": "4274",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 2
            },
            {
                "ref": "6558",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 4
            },
            {
                "ref": "3024",
                "color_code": "103",
                "color_hex": "#FFEC6C",
                "color_name": "Bright Light Yellow",
                "qty": 4
            },
            {
                "ref": "3023",
                "color_code": "103",
                "color_hex": "#FFEC6C",
                "color_name": "Bright Light Yellow",
                "qty": 4
            },
            {
                "ref": "32952",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "87620",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "2489",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "4740",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "30553",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "2429c01",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3639",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "44300",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "30132",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3023",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 10
            },
            {
                "ref": "3623",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3710",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 26
            },
            {
                "ref": "3460",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3020",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 7
            },
            {
                "ref": "15573",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "32028",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 6
            },
            {
                "ref": "11458",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 28
            },
            {
                "ref": "4073",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 35
            },
            {
                "ref": "15392",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "15301c01",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "61409",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 6
            },
            {
                "ref": "85984",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "85984pb127",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "15068",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "32209",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "10197",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "6538c",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "32064",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3894",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "32530",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "15712",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 3
            },
            {
                "ref": "43723",
                "color_code": "85",
                "color_hex": "#6B6D67",
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
                "color_code": "69",
                "color_hex": "#948972",
                "color_name": "Dark Tan",
                "qty": 1
            },
            {
                "ref": "3023",
                "color_code": "39",
                "color_hex": "#00828E",
                "color_name": "Dark Turquoise",
                "qty": 2
            },
            {
                "ref": "4032",
                "color_code": "39",
                "color_hex": "#00828E",
                "color_name": "Dark Turquoise",
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
                "color_hex": "#24793D",
                "color_name": "Green",
                "qty": 1
            },
            {
                "ref": "2714b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "99781",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "18671",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3004",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 5
            },
            {
                "ref": "3010",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3003",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "87087",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "22885",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "4589b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "41531",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3937",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3938",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "60849",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "18738",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "99563",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3024",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3023",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6
            },
            {
                "ref": "3460",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "2420",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 12
            },
            {
                "ref": "3021",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3031",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "92280",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "18677",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "87580",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3176",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "11477",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 12
            },
            {
                "ref": "2341",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "60219",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "4599b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "4265c",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 12
            },
            {
                "ref": "4519",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "32073",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "44294",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "57585",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "32039",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "6541",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "41677",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "11478",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "32054",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "62462",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3673",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3738",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3069",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "2431",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "43712",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "43719",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3795",
                "color_code": "34",
                "color_hex": "#D6EF00",
                "color_name": "Lime",
                "qty": 1
            },
            {
                "ref": "34103",
                "color_code": "42",
                "color_hex": "#7396C8",
                "color_name": "Medium Blue",
                "qty": 2
            },
            {
                "ref": "64567",
                "color_code": "67",
                "color_hex": "#A4A8B3",
                "color_name": "Metallic Silver",
                "qty": 1
            },
            {
                "ref": "96874",
                "color_code": "4",
                "color_hex": "#CF650F",
                "color_name": "Orange",
                "qty": 1
            },
            {
                "ref": "3003",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1
            },
            {
                "ref": "32952",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2
            },
            {
                "ref": "30414",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2
            },
            {
                "ref": "3062",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1
            },
            {
                "ref": "3710",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2
            },
            {
                "ref": "3020",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 3
            },
            {
                "ref": "3795",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 3
            },
            {
                "ref": "32062",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 4
            },
            {
                "ref": "3707",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1
            },
            {
                "ref": "32530",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1
            },
            {
                "ref": "3005",
                "color_code": "55",
                "color_hex": "#587083",
                "color_name": "Sand Blue",
                "qty": 4
            },
            {
                "ref": "3070",
                "color_code": "55",
                "color_hex": "#587083",
                "color_name": "Sand Blue",
                "qty": 11
            },
            {
                "ref": "3069",
                "color_code": "55",
                "color_hex": "#587083",
                "color_name": "Sand Blue",
                "qty": 1
            },
            {
                "ref": "11211",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 2
            },
            {
                "ref": "3023",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 3
            },
            {
                "ref": "3623",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 4
            },
            {
                "ref": "4477",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 1
            },
            {
                "ref": "3022",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 1
            },
            {
                "ref": "2854",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 1
            },
            {
                "ref": "3749",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 1
            },
            {
                "ref": "6541",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 2
            },
            {
                "ref": "32002",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 2
            },
            {
                "ref": "21849pb04",
                "color_code": "12",
                "color_hex": "#FEFEFE",
                "color_name": "Trans-Clear",
                "qty": 1
            },
            {
                "ref": "4589b",
                "color_code": "50",
                "color_hex": "#FB96AB",
                "color_name": "Trans-Dark Pink",
                "qty": 4
            },
            {
                "ref": "30374",
                "color_code": "15",
                "color_hex": "#AEE9EF",
                "color_name": "Trans-Light Blue",
                "qty": 1
            },
            {
                "ref": "4073",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 4
            },
            {
                "ref": "15303",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 6
            },
            {
                "ref": "63965",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "3005",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 3
            },
            {
                "ref": "3622",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "4216",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "30414",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "4740",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "2429c01",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 3
            },
            {
                "ref": "3023",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 8
            },
            {
                "ref": "3623",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "3710",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 5
            },
            {
                "ref": "3666",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 5
            },
            {
                "ref": "3460",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "2420",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "3021",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1
            },
            {
                "ref": "3020",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 3
            },
            {
                "ref": "3795",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "3832",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "2445",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "2639",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "3958",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "3839b",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1
            },
            {
                "ref": "92593",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 3
            },
            {
                "ref": "85861",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "15403",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "x71",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "54200",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "85984",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 8
            },
            {
                "ref": "4286",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "3297",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "3040",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "92946",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "28192",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 12
            },
            {
                "ref": "15068",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "93273",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "85970",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "3676",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "4871",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "32000",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "6632",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "3070",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "3069",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1
            },
            {
                "ref": "63864",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "4162",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "3068",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1
            },
            {
                "ref": "14719",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "26603",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 6
            },
            {
                "ref": "87079",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "6179",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "27263",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "43713",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "43723",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "43722",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 3
            },
            {
                "ref": "14181",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "30355",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "30356",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "20309",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 12
            },
            {
                "ref": "6070",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "15573",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 1
            },
            {
                "ref": "4265c",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 3
            },
            {
                "ref": "32064",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2
            }
        ]
    },
    "75337-1": {
        "name": "AT-TE Walker",
        "minifigures": [
            {
                "ref": "sw1194",
                "name": "Ahsoka Tano",
                "qty": 1
            },
            {
                "ref": "sw1195",
                "name": "Barriss Offee",
                "qty": 1
            },
            {
                "ref": "sw1093",
                "name": "Clone Trooper, 501st Legion",
                "qty": 6
            },
            {
                "ref": "sw1196",
                "name": "Clone Commander Bly",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "75337stk01",
                "color_code": "0",
                "color_hex": "#808080",
                "color_name": "Various",
                "qty": 1
            },
            {
                "ref": "4592c02",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "28802",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "11215",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 6
            },
            {
                "ref": "3245c",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "2877",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 3
            },
            {
                "ref": "4589b",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 10
            },
            {
                "ref": "6154",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "30383",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 8
            },
            {
                "ref": "44302",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 8
            },
            {
                "ref": "60471",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "37762",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "57899",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "92738",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "58247",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "3023",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "3031",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "3958",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "3036",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "78257",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "2817",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "99206",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "64799",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "4073",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "49307",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "32802",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "4871",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "3713",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "32062",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "3705",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "32015",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "6536",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "6538c",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "26287",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 3
            },
            {
                "ref": "6541",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "32064",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "3702",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "3743",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "32523",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "32524",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1
            },
            {
                "ref": "40490",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "41677",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 8
            },
            {
                "ref": "32054",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 3
            },
            {
                "ref": "32333",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 37
            },
            {
                "ref": "43093",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 6
            },
            {
                "ref": "6558",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 16
            },
            {
                "ref": "78258",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "62113",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "99780",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 5
            },
            {
                "ref": "93274",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 3
            },
            {
                "ref": "3010",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3009",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3003",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3001",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "87087",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "2877",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 3
            },
            {
                "ref": "87081",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "61780",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "44359",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 12
            },
            {
                "ref": "44301b",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "44302",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 6
            },
            {
                "ref": "3710",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 10
            },
            {
                "ref": "3666",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "60479",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3022",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "2420",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3021",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3020",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 5
            },
            {
                "ref": "3795",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3034",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3031",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3032",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 6
            },
            {
                "ref": "3036",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "15573",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 13
            },
            {
                "ref": "32028",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "92107",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "4073",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 12
            },
            {
                "ref": "4032",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 12
            },
            {
                "ref": "60474",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 6
            },
            {
                "ref": "69755",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "61409",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "85984pb127",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3045",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3678b",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "6091",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "15068",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "93606",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "4287",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3660",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 3
            },
            {
                "ref": "32209",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 5
            },
            {
                "ref": "32013",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 6
            },
            {
                "ref": "42003",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "6541",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "32000",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 10
            },
            {
                "ref": "3701",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "32018",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3648",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 6
            },
            {
                "ref": "18654",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "41677",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "6629",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "15712",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2
            },
            {
                "ref": "2412b",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 10
            },
            {
                "ref": "6179",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "98138",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "27925",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 8
            },
            {
                "ref": "47759",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1
            },
            {
                "ref": "51739",
                "color_code": "85",
                "color_hex": "#6B6D67",
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
                "color_code": "69",
                "color_hex": "#948972",
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
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6
            },
            {
                "ref": "63965",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6
            },
            {
                "ref": "36840",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "99781",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3005",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3004",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3001",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "11211",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "2653",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 3
            },
            {
                "ref": "44358",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6
            },
            {
                "ref": "43898",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6
            },
            {
                "ref": "3937",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "6134",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "53923",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "44567b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "44302",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "92582",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "87544",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "91501",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "87421",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3023",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 12
            },
            {
                "ref": "3623",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3710",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "4477",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "2420",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "3021",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3020",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3795",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "60470b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "63868",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "10247",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "2476",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "4073",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 8
            },
            {
                "ref": "2654",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "60474",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 11
            },
            {
                "ref": "11213",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 10
            },
            {
                "ref": "69754",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "54200",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 8
            },
            {
                "ref": "85984",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "4286",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 8
            },
            {
                "ref": "3297",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3040",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 12
            },
            {
                "ref": "3049c",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "15571",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "92946",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "37352",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "11477",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 3
            },
            {
                "ref": "30165",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 3
            },
            {
                "ref": "15068",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 29
            },
            {
                "ref": "6081",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "85970",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "4287",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 8
            },
            {
                "ref": "3747b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "3665",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "4871",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "2449",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "4599b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "4265c",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "32073",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "32184",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3700",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6
            },
            {
                "ref": "3701",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "32531",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "73109",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "32316",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "41677",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "60484",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "4274",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 18
            },
            {
                "ref": "3673",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3069",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "63864",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 8
            },
            {
                "ref": "2431",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6
            },
            {
                "ref": "26603",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "87079",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "2432",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "33909",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "14769",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "27925",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 40
            },
            {
                "ref": "61485",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "47755",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "48933",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "52031",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "50955",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "50956",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "26601",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "51739",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "43723",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "43722",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "2419",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "78443",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "78444",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2
            },
            {
                "ref": "54384",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "54383",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "6106",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            },
            {
                "ref": "50305",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "50304",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1
            },
            {
                "ref": "96874",
                "color_code": "4",
                "color_hex": "#CF650F",
                "color_name": "Orange",
                "qty": 1
            },
            {
                "ref": "61190d",
                "color_code": "4",
                "color_hex": "#CF650F",
                "color_name": "Orange",
                "qty": 4
            },
            {
                "ref": "61190c",
                "color_code": "4",
                "color_hex": "#CF650F",
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
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1
            },
            {
                "ref": "44865",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1
            },
            {
                "ref": "3062",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1
            },
            {
                "ref": "3020",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 10
            },
            {
                "ref": "3660",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2
            },
            {
                "ref": "32062",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 11
            },
            {
                "ref": "3705",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1
            },
            {
                "ref": "3707",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2
            },
            {
                "ref": "62462",
                "color_code": "88",
                "color_hex": "#5F3109",
                "color_name": "Reddish Brown",
                "qty": 4
            },
            {
                "ref": "4079",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 1
            },
            {
                "ref": "78329",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 2
            },
            {
                "ref": "3022",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 1
            },
            {
                "ref": "78256",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 1
            },
            {
                "ref": "4032",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 10
            },
            {
                "ref": "49307",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 2
            },
            {
                "ref": "3749",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 4
            },
            {
                "ref": "3700",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 1
            },
            {
                "ref": "33909",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 7
            },
            {
                "ref": "4073",
                "color_code": "108",
                "color_hex": "#56E646",
                "color_name": "Trans-Bright Green",
                "qty": 4
            },
            {
                "ref": "87544",
                "color_code": "12",
                "color_hex": "#FEFEFE",
                "color_name": "Trans-Clear",
                "qty": 4
            },
            {
                "ref": "4073",
                "color_code": "12",
                "color_hex": "#FEFEFE",
                "color_name": "Trans-Clear",
                "qty": 4
            },
            {
                "ref": "3023",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 11
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 6
            },
            {
                "ref": "4073",
                "color_code": "98",
                "color_hex": "#EF8E1B",
                "color_name": "Trans-Orange",
                "qty": 4
            },
            {
                "ref": "99781",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1
            },
            {
                "ref": "30304",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1
            },
            {
                "ref": "4085d",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2
            },
            {
                "ref": "4265c",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 1
            },
            {
                "ref": "4519",
                "color_code": "3",
                "color_hex": "#F4CC2E",
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
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 4
            },
            {
                "ref": "3002",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 4
            },
            {
                "ref": "3003",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 4
            },
            {
                "ref": "3004",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 6
            },
            {
                "ref": "3005",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 10
            },
            {
                "ref": "3010",
                "color_code": "6",
                "color_hex": "#24793D",
                "color_name": "Green",
                "qty": 4
            },
            {
                "ref": "3009",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2
            },
            {
                "ref": "3008",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 2
            },
            {
                "ref": "3020",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 6
            },
            {
                "ref": "3021",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 6
            },
            {
                "ref": "3022",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 8
            },
            {
                "ref": "3023",
                "color_code": "6",
                "color_hex": "#24793D",
                "color_name": "Green",
                "qty": 12
            },
            {
                "ref": "3024",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 10
            },
            {
                "ref": "3034",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 2
            },
            {
                "ref": "3035",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2
            },
            {
                "ref": "3031",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4
            },
            {
                "ref": "2420",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "3068",
                "color_code": "6",
                "color_hex": "#24793D",
                "color_name": "Green",
                "qty": 4
            },
            {
                "ref": "3069",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 6
            },
            {
                "ref": "3070",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 8
            },
            {
                "ref": "6636",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 4
            },
            {
                "ref": "4162",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2
            },
            {
                "ref": "3038",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "3039",
                "color_code": "6",
                "color_hex": "#24793D",
                "color_name": "Green",
                "qty": 6
            },
            {
                "ref": "3040",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 6
            },
            {
                "ref": "3298",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 4
            },
            {
                "ref": "3037",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 4
            },
            {
                "ref": "2412",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 8
            },
            {
                "ref": "3710",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 10
            },
            {
                "ref": "3622",
                "color_code": "6",
                "color_hex": "#24793D",
                "color_name": "Green",
                "qty": 4
            },
            {
                "ref": "3666",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 6
            },
            {
                "ref": "3795",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 4
            },
            {
                "ref": "4073",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 12
            },
            {
                "ref": "3062",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 6
            },
            {
                "ref": "22885",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 4
            },
            {
                "ref": "32000",
                "color_code": "6",
                "color_hex": "#24793D",
                "color_name": "Green",
                "qty": 4
            },
            {
                "ref": "3700",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 4
            },
            {
                "ref": "2877",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 6
            },
            {
                "ref": "3659",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2
            },
            {
                "ref": "6141",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 12
            },
            {
                "ref": "15573",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 8
            },
            {
                "ref": "14719",
                "color_code": "6",
                "color_hex": "#24793D",
                "color_name": "Green",
                "qty": 4
            },
            {
                "ref": "18674",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 4
            },
            {
                "ref": "32013",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 4
            },
            {
                "ref": "6536",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 4
            },
            {
                "ref": "4274",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 10
            },
            {
                "ref": "3673",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 10
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 15
            },
            {
                "ref": "3705",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4
            },
            {
                "ref": "3003",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4
            }
        ]
    },
    "75038-1": {
        "name": "Jedi Interceptor",
        "minifigures": [
            {
                "ref": "sw0526",
                "name": "Anakin Skywalker (Dark Brown Legs, Headset)",
                "qty": 1
            },
            {
                "ref": "sw0527",
                "name": "Astromech Droid, R2-D2 - Flat Silver Head, Red Dots and Small Receptor",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "75038stk01a",
                "color_code": "0",
                "color_hex": "",
                "color_name": "",
                "qty": 1,
                "name": "Sticker Sheet for Set 75038 - International Version - (16334/6058264)"
            },
            {
                "ref": "63965",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Bar   6L with Stop Ring"
            },
            {
                "ref": "4589b",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Cone 1 x 1 with Top Groove"
            },
            {
                "ref": "64567",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Minifigure, Weapon Lightsaber Hilt Straight"
            },
            {
                "ref": "3020",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Plate 2 x 4"
            },
            {
                "ref": "3795",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Plate 2 x 6"
            },
            {
                "ref": "3832",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Plate 2 x 10"
            },
            {
                "ref": "2540",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate, Modified 1 x 2 with Bar Handle on Side - Free Ends"
            },
            {
                "ref": "63868",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Plate, Modified 1 x 2 with Clip on End (Horizontal Grip)"
            },
            {
                "ref": "41678",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Technic, Axle and Pin Connector Perpendicular Double Split"
            },
            {
                "ref": "6538c",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Axle Connector 2L (Smooth with x Hole + Orientation)"
            },
            {
                "ref": "32054",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Pin 3L with Friction Ridges and Stop Bush"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Pin with Short Friction Ridges"
            },
            {
                "ref": "4274",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1,
                "name": "Technic, Pin 1/2 without Friction Ridges"
            },
            {
                "ref": "3002",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Brick 2 x 3"
            },
            {
                "ref": "44570",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Hinge Plate 3 x 4 Locking Dual 2 Fingers"
            },
            {
                "ref": "3623",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Plate 1 x 3"
            },
            {
                "ref": "3020",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 8,
                "name": "Plate 2 x 4"
            },
            {
                "ref": "3832",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Plate 2 x 10"
            },
            {
                "ref": "87580",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Plate, Modified 2 x 2 with Groove and 1 Stud in Center (Jumper)"
            },
            {
                "ref": "15301c01",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Projectile Launcher, 1 x 4 Spring Shooter with Light Bluish Gray Top"
            },
            {
                "ref": "72454",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Slope, Inverted 45 4 x 4 Double with 2 Holes"
            },
            {
                "ref": "6536",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Technic, Axle and Pin Connector Perpendicular"
            },
            {
                "ref": "32000",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Technic, Brick 1 x 2 with Holes"
            },
            {
                "ref": "3070",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Tile 1 x 1"
            },
            {
                "ref": "87079",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Tile 2 x 4"
            },
            {
                "ref": "3710",
                "color_code": "69",
                "color_hex": "#948972",
                "color_name": "Dark Tan",
                "qty": 18,
                "name": "Plate 1 x 4"
            },
            {
                "ref": "6587",
                "color_code": "69",
                "color_hex": "#948972",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Technic, Axle  3L with Stud"
            },
            {
                "ref": "2412b",
                "color_code": "297",
                "color_hex": "#899395",
                "color_name": "Flat Silver",
                "qty": 4,
                "name": "Tile, Modified 1 x 2 Grille with Bottom Groove"
            },
            {
                "ref": "30374",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Bar   4L (Lightsaber Blade / Wand)"
            },
            {
                "ref": "30526",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Brick, Modified 1 x 2 with Pins"
            },
            {
                "ref": "44567a",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Hinge Plate 1 x 2 Locking with 1 Finger on Side with Bottom Groove"
            },
            {
                "ref": "60471",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Hinge Plate 1 x 2 Locking with 2 Fingers on Side"
            },
            {
                "ref": "3666",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Plate 1 x 6"
            },
            {
                "ref": "4477",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Plate 1 x 10"
            },
            {
                "ref": "3022",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Plate 2 x 2"
            },
            {
                "ref": "3021",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 7,
                "name": "Plate 2 x 3"
            },
            {
                "ref": "3795",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Plate 2 x 6"
            },
            {
                "ref": "4081b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Plate, Modified 1 x 1 with Light Attachment - Thick Ring"
            },
            {
                "ref": "4085d",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 10,
                "name": "Plate, Modified 1 x 1 with Open O Clip Thick (Vertical Grip)"
            },
            {
                "ref": "3794a",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Plate, Modified 1 x 2 with 1 Stud without Groove (Jumper)"
            },
            {
                "ref": "2654",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Plate, Round 2 x 2 with Rounded Bottom (Boat Stud)"
            },
            {
                "ref": "3747b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Slope, Inverted 33 3 x 2 with Flat Bottom Pin and Connections between Studs"
            },
            {
                "ref": "4871",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Slope, Inverted 45 4 x 2 Double with 2 x 2 Cutout"
            },
            {
                "ref": "32064",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Brick 1 x 2 with Axle Hole"
            },
            {
                "ref": "3700",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Technic, Brick 1 x 2 with Hole"
            },
            {
                "ref": "2431",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Tile 1 x 4"
            },
            {
                "ref": "3068",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Tile 2 x 2"
            },
            {
                "ref": "87079",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Tile 2 x 4"
            },
            {
                "ref": "98138pb020",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Tile, Round 1 x 1 with SW Emblem of the Galactic Republic with 6 Spokes Pattern"
            },
            {
                "ref": "43723",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Wedge, Plate 3 x 2 Left"
            },
            {
                "ref": "43722",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Wedge, Plate 3 x 2 Right"
            },
            {
                "ref": "30503",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Wedge, Plate 4 x 4 Cut Corner"
            },
            {
                "ref": "55981",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Wheel 18mm D. x 14mm with Pin Hole, Fake Bolts and Shallow Spokes"
            },
            {
                "ref": "64567",
                "color_code": "67",
                "color_hex": "#A4A8B3",
                "color_name": "Metallic Silver",
                "qty": 1,
                "name": "Minifigure, Weapon Lightsaber Hilt Straight"
            },
            {
                "ref": "4073",
                "color_code": "115",
                "color_hex": "#CB9B2A",
                "color_name": "Pearl Gold",
                "qty": 12,
                "name": "Plate, Round 1 x 1"
            },
            {
                "ref": "32062",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 3,
                "name": "Technic, Axle  2L Notched"
            },
            {
                "ref": "6091",
                "color_code": "88",
                "color_hex": "#5F3109",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Slope, Curved 2 x 1 x 1 1/3 with Recessed Stud"
            },
            {
                "ref": "61678",
                "color_code": "88",
                "color_hex": "#5F3109",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Slope, Curved 4 x 1"
            },
            {
                "ref": "3068",
                "color_code": "88",
                "color_hex": "#5F3109",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Tile 2 x 2"
            },
            {
                "ref": "3022",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 3,
                "name": "Plate 2 x 2"
            },
            {
                "ref": "15303",
                "color_code": "108",
                "color_hex": "#56E646",
                "color_name": "Trans-Bright Green",
                "qty": 3,
                "name": "Projectile Arrow, Bar 8L with Round End (Spring Shooter Dart)"
            },
            {
                "ref": "3960pb013",
                "color_code": "13",
                "color_hex": "#625E51",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Dish 4 x 4 Inverted (Radar) with Solid Stud with Radial Jedi Starfighter / TIE Cockpit Pattern"
            },
            {
                "ref": "10312pb01",
                "color_code": "13",
                "color_hex": "#625E51",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Windscreen 10 x 6 x 3 Bubble Canopy Double Tapered with Square Front Cutout with Light Bluish Gray Jedi Starfighter, Handle and Two Red Triangles on Top Pattern"
            },
            {
                "ref": "30374",
                "color_code": "15",
                "color_hex": "#AEE9EF",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Bar   4L (Lightsaber Blade / Wand)"
            },
            {
                "ref": "4740",
                "color_code": "15",
                "color_hex": "#AEE9EF",
                "color_name": "Trans-Light Blue",
                "qty": 2,
                "name": "Dish 2 x 2 Inverted (Radar)"
            },
            {
                "ref": "2877",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Brick, Modified 1 x 2 with Grille / Fluted Profile"
            },
            {
                "ref": "x1435",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Flag 5 x 6 Hexagonal"
            },
            {
                "ref": "3023",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Plate 1 x 2"
            },
            {
                "ref": "3666",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Plate 1 x 6"
            },
            {
                "ref": "2420",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Plate 2 x 2 Corner"
            },
            {
                "ref": "3021",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Plate 2 x 3"
            },
            {
                "ref": "3298",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Slope 33 3 x 2"
            },
            {
                "ref": "2431",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Tile 1 x 4"
            },
            {
                "ref": "2412b",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Tile, Modified 1 x 2 Grille with Bottom Groove"
            },
            {
                "ref": "43710",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Wedge 4 x 2 Triple Left"
            },
            {
                "ref": "43711",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Wedge 4 x 2 Triple Right"
            },
            {
                "ref": "51739",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Wedge, Plate 2 x 4"
            },
            {
                "ref": "50305",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Wedge, Plate 8 x 3 Pentagonal Left"
            },
            {
                "ref": "50304",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Wedge, Plate 8 x 3 Pentagonal Right"
            }
        ]
    },
    "31062-1": {
        "name": "Robo Explorer",
        "minifigures": [],
        "parts": [
            {
                "ref": "44728",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Bracket 1 x 2 - 2 x 2"
            },
            {
                "ref": "3956",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Bracket 2 x 2 - 2 x 2 with 2 Holes"
            },
            {
                "ref": "30340",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Minifigure, Utensil Flotation Ring (Life Preserver)"
            },
            {
                "ref": "3710",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate 1 x 4"
            },
            {
                "ref": "3022",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Plate 2 x 2"
            },
            {
                "ref": "48336",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate, Modified 1 x 2 with Bar Handle on Side - Closed Ends"
            },
            {
                "ref": "50949",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Plate, Modified 1 x 2 with Racers Car Grille"
            },
            {
                "ref": "4032",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 3,
                "name": "Plate, Round 2 x 2 with Axle Hole"
            },
            {
                "ref": "54200",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Slope 30 1 x 1 x 2/3"
            },
            {
                "ref": "85984",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Slope 30 1 x 2 x 2/3"
            },
            {
                "ref": "11477",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Slope, Curved 2 x 1 x 2/3"
            },
            {
                "ref": "15068",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Slope, Curved 2 x 2 x 2/3"
            },
            {
                "ref": "93273",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Slope, Curved 4 x 1 x 2/3 Double"
            },
            {
                "ref": "47455",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Technic Rotation Joint Ball Loop with 2 Perpendicular Pins with Friction"
            },
            {
                "ref": "32064",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Brick 1 x 2 with Axle Hole"
            },
            {
                "ref": "3894",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Brick 1 x 6 with Holes"
            },
            {
                "ref": "92013",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Technic, Brick Modified 2 x 2 with Ball Socket and Axle Hole - Straight Forks with Round Ends and Open Sides"
            },
            {
                "ref": "3873",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 40,
                "name": "Technic, Link Tread"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Pin with Short Friction Ridges"
            },
            {
                "ref": "3069",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Tile 1 x 2"
            },
            {
                "ref": "61254",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Tire 24mm D. x 7mm Offset Tread - Band Around Center of Tread"
            },
            {
                "ref": "50745",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Vehicle, Mudguard 4 x 2 1/2 x 1 2/3 with Arch Round"
            },
            {
                "ref": "11291",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Wedge 3 x 4 x 2/3 Curved with Cutout"
            },
            {
                "ref": "3003",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Brick 2 x 2"
            },
            {
                "ref": "553",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Brick, Round 2 x 2 Dome Top"
            },
            {
                "ref": "54930c02",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Electric, Light Brick 2 x 3 x 1 1/3 with Trans-Clear Top and Yellow LED Light (Glows Orange)"
            },
            {
                "ref": "3023",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Plate 1 x 2"
            },
            {
                "ref": "2420",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Plate 2 x 2 Corner"
            },
            {
                "ref": "15573",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Plate, Modified 1 x 2 with 1 Stud with Groove and Bottom Stud Holder (Jumper)"
            },
            {
                "ref": "2654",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Plate, Round 2 x 2 with Rounded Bottom (Boat Stud)"
            },
            {
                "ref": "3040",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Slope 45 2 x 1"
            },
            {
                "ref": "55013",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Technic, Axle  8L with Stop"
            },
            {
                "ref": "17114",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Technic, Brick Modified 2 x 2 with 2 Ball Joints and Axle Hole"
            },
            {
                "ref": "57909b",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Technic, Brick Modified 2 x 2 with Ball Joint and Axle Hole with 6 Holes in Ball"
            },
            {
                "ref": "92013",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Technic, Brick Modified 2 x 2 with Ball Socket and Axle Hole - Straight Forks with Round Ends and Open Sides"
            },
            {
                "ref": "48170",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Technic, Brick Modified 2 x 2 with Pin Hole and Rotation Joint Ball Half Horizontal"
            },
            {
                "ref": "48169",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Technic, Brick Modified 2 x 2 with Pin Hole and Rotation Joint Socket"
            },
            {
                "ref": "32530",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Technic, Pin Connector Plate 1 x 2 x 1 2/3 with 2 Holes on Top"
            },
            {
                "ref": "3709",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Technic, Plate 2 x 4 with 3 Holes"
            },
            {
                "ref": "11211",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Brick, Modified 1 x 2 with Studs on 1 Side"
            },
            {
                "ref": "87620",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Brick, Modified Facet 2 x 2"
            },
            {
                "ref": "2335",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Flag 2 x 2 Square with Flat Edge"
            },
            {
                "ref": "3024",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 10,
                "name": "Plate 1 x 1"
            },
            {
                "ref": "3623",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Plate 1 x 3"
            },
            {
                "ref": "3666",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Plate 1 x 6"
            },
            {
                "ref": "60470b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Plate, Modified 1 x 2 with 2 Open O Clips (Horizontal Grip)"
            },
            {
                "ref": "2540",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Plate, Modified 1 x 2 with Bar Handle on Side - Free Ends"
            },
            {
                "ref": "14704",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Plate, Modified 1 x 2 with Small Tow Ball Socket on Side"
            },
            {
                "ref": "4073",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Plate, Round 1 x 1"
            },
            {
                "ref": "24201",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Slope, Curved 2 x 1 x 2/3 Inverted"
            },
            {
                "ref": "13547",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Slope, Curved 4 x 1 Inverted"
            },
            {
                "ref": "93273",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Slope, Curved 4 x 1 x 2/3 Double"
            },
            {
                "ref": "2850b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Technic Engine Cylinder with Partial Hollow Studs on Top and Bottom Slots"
            },
            {
                "ref": "2736",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Axle  1L with Tow Ball"
            },
            {
                "ref": "2723",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Technic, Disk 3 x 3"
            },
            {
                "ref": "94925",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Gear 16 Tooth - Axle Hole with Closed Sides"
            },
            {
                "ref": "13548",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Wedge 2 x 2 (Slope 45 Corner)"
            },
            {
                "ref": "41770",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Wedge, Plate 4 x 2 Left"
            },
            {
                "ref": "41769",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Wedge, Plate 4 x 2 Right"
            },
            {
                "ref": "30503",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Wedge, Plate 4 x 4 Cut Corner"
            },
            {
                "ref": "3005",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 2,
                "name": "Brick 1 x 1"
            },
            {
                "ref": "3020",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 1,
                "name": "Plate 2 x 4"
            },
            {
                "ref": "54200",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 2,
                "name": "Slope 30 1 x 1 x 2/3"
            },
            {
                "ref": "3039",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 1,
                "name": "Slope 45 2 x 2"
            },
            {
                "ref": "6191",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 1,
                "name": "Slope, Curved 1 x 4 x 1 1/3"
            },
            {
                "ref": "6091",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 4,
                "name": "Slope, Curved 2 x 1 x 1 1/3 with Recessed Stud"
            },
            {
                "ref": "50950",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 2,
                "name": "Slope, Curved 3 x 1"
            },
            {
                "ref": "6215",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 1,
                "name": "Slope, Curved 3 x 2 with 4 Studs"
            },
            {
                "ref": "61678",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 2,
                "name": "Slope, Curved 4 x 1"
            },
            {
                "ref": "93273",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 1,
                "name": "Slope, Curved 4 x 1 x 2/3 Double"
            },
            {
                "ref": "3069",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 2,
                "name": "Tile 1 x 2"
            },
            {
                "ref": "2431",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 1,
                "name": "Tile 1 x 4"
            },
            {
                "ref": "3068",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 1,
                "name": "Tile 2 x 2"
            },
            {
                "ref": "50745",
                "color_code": "156",
                "color_hex": "#68C3E2",
                "color_name": "Medium Azure",
                "qty": 2,
                "name": "Vehicle, Mudguard 4 x 2 1/2 x 1 2/3 with Arch Round"
            },
            {
                "ref": "3713",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 5,
                "name": "Technic Bush"
            },
            {
                "ref": "32062",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Technic, Axle  2L Notched"
            },
            {
                "ref": "24316",
                "color_code": "88",
                "color_hex": "#5F3109",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Technic, Axle  3L with Stop"
            },
            {
                "ref": "15462",
                "color_code": "88",
                "color_hex": "#5F3109",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Technic, Axle  5L with Stop"
            },
            {
                "ref": "4070",
                "color_code": "12",
                "color_hex": "#FEFEFE",
                "color_name": "Trans-Clear",
                "qty": 2,
                "name": "Brick, Modified 1 x 1 with Headlight"
            },
            {
                "ref": "98138",
                "color_code": "20",
                "color_hex": "#227740",
                "color_name": "Trans-Green",
                "qty": 2,
                "name": "Tile, Round 1 x 1"
            },
            {
                "ref": "4073",
                "color_code": "16",
                "color_hex": "#BFFE00",
                "color_name": "Trans-Neon Green",
                "qty": 2,
                "name": "Plate, Round 1 x 1"
            },
            {
                "ref": "54200",
                "color_code": "16",
                "color_hex": "#BFFE00",
                "color_name": "Trans-Neon Green",
                "qty": 3,
                "name": "Slope 30 1 x 1 x 2/3"
            },
            {
                "ref": "4073",
                "color_code": "98",
                "color_hex": "#EF8E1B",
                "color_name": "Trans-Orange",
                "qty": 2,
                "name": "Plate, Round 1 x 1"
            },
            {
                "ref": "2436b",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Bracket 1 x 2 - 1 x 4 with Rounded Corners"
            },
            {
                "ref": "2412b",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Tile, Modified 1 x 2 Grille with Bottom Groove"
            },
            {
                "ref": "56903",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Wheel 18mm D. x  8mm with Fake Bolts and Shallow Spokes and Axle Hole"
            },
            {
                "ref": "4265c",
                "color_code": "3",
                "color_hex": "#F4CC2E",
                "color_name": "Yellow",
                "qty": 3,
                "name": "Technic Bush 1/2 Smooth"
            }
        ]
    },
    "75018-1": {
        "name": "Jek-14",
        "minifigures": [
            {
                "ref": "sw0477",
                "name": "Astromech Droid, R4-G0",
                "qty": 1
            },
            {
                "ref": "sw0476",
                "name": "Bounty Hunter",
                "qty": 1
            },
            {
                "ref": "sw0475",
                "name": "Jek-14 - Clone Helmet",
                "qty": 1
            },
            {
                "ref": "sw0478",
                "name": "Special Forces Clone Trooper",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "75018stk01",
                "color_code": "0",
                "color_hex": "",
                "color_name": "",
                "qty": 1,
                "name": "Sticker Sheet for Set 75018 - (14559/6040509)"
            },
            {
                "ref": "4592c02",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Antenna Small Base with Black Lever (4592 / 4593)"
            },
            {
                "ref": "99781",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 6,
                "name": "Bracket 1 x 2 - 1 x 2"
            },
            {
                "ref": "4070",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Brick, Modified 1 x 1 with Headlight"
            },
            {
                "ref": "2877",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 3,
                "name": "Brick, Modified 1 x 2 with Grille / Fluted Profile"
            },
            {
                "ref": "4742",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Cone 4 x 4 x 2 Hollow No Studs"
            },
            {
                "ref": "87617",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Cylinder 1 x 5 1/2 with Bar Handle (Friction Cylinder)"
            },
            {
                "ref": "43121",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Engine, Large"
            },
            {
                "ref": "30554b",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Hinge Cylinder 1 x 3 Locking with 1 Finger and 2 Fingers on Ends with Hole"
            },
            {
                "ref": "98285",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Hinge Plate 2 x 4 with Pin Hole and 3 Holes - Bottom"
            },
            {
                "ref": "92081",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Minifigure, Hair Combed Front to Rear"
            },
            {
                "ref": "57899",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Minifigure, Weapon Gun, Blaster SW Long"
            },
            {
                "ref": "58247",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Minifigure, Weapon Gun, Blaster SW Standard"
            },
            {
                "ref": "4865b",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Panel 1 x 2 x 1 with Rounded Corners"
            },
            {
                "ref": "3024",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate 1 x 1"
            },
            {
                "ref": "3023",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate 1 x 2"
            },
            {
                "ref": "3623",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate 1 x 3"
            },
            {
                "ref": "3710",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 5,
                "name": "Plate 1 x 4"
            },
            {
                "ref": "3022",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate 2 x 2"
            },
            {
                "ref": "2420",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate 2 x 2 Corner"
            },
            {
                "ref": "3832",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 5,
                "name": "Plate 2 x 10"
            },
            {
                "ref": "3029",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate 4 x 12"
            },
            {
                "ref": "3958",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Plate 6 x 6"
            },
            {
                "ref": "3794b",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate, Modified 1 x 2 with 1 Stud with Groove (Jumper)"
            },
            {
                "ref": "60470b",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Plate, Modified 1 x 2 with 2 Open O Clips (Horizontal Grip)"
            },
            {
                "ref": "48336",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 10,
                "name": "Plate, Modified 1 x 2 with Bar Handle on Side - Closed Ends"
            },
            {
                "ref": "2817",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 6,
                "name": "Plate, Modified 2 x 2 with Pin Holes"
            },
            {
                "ref": "3176",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Plate, Modified 2 x 3 with Hole"
            },
            {
                "ref": "4073",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Plate, Round 1 x 1"
            },
            {
                "ref": "54200",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Slope 30 1 x 1 x 2/3"
            },
            {
                "ref": "85984",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 9,
                "name": "Slope 30 1 x 2 x 2/3"
            },
            {
                "ref": "92946",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Slope 45 2 x 1 with 2/3 Cutout"
            },
            {
                "ref": "3039",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Slope 45 2 x 2"
            },
            {
                "ref": "3037",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Slope 45 2 x 4"
            },
            {
                "ref": "60481",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 6,
                "name": "Slope 65 2 x 1 x 2"
            },
            {
                "ref": "61678",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 6,
                "name": "Slope, Curved 4 x 1"
            },
            {
                "ref": "85970",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Slope, Curved 10 x 1"
            },
            {
                "ref": "3747b",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Slope, Inverted 33 3 x 2 with Flat Bottom Pin and Connections between Studs"
            },
            {
                "ref": "3660",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Slope, Inverted 45 2 x 2 with Flat Bottom Pin"
            },
            {
                "ref": "3707",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Axle  8L"
            },
            {
                "ref": "3708",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Axle 12L"
            },
            {
                "ref": "41678",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Technic, Axle and Pin Connector Perpendicular Double Split"
            },
            {
                "ref": "63869",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Technic, Axle and Pin Connector Perpendicular Triple"
            },
            {
                "ref": "6538c",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Technic, Axle Connector 2L (Smooth with x Hole + Orientation)"
            },
            {
                "ref": "3894",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 5,
                "name": "Technic, Brick 1 x 6 with Holes"
            },
            {
                "ref": "32531",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Technic, Brick 4 x 6 Open Center"
            },
            {
                "ref": "3743",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Technic, Gear Rack 1 x 4"
            },
            {
                "ref": "32140",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Liftarm, Modified Bent Thick L-Shape 2 x 4"
            },
            {
                "ref": "33299",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Technic, Liftarm, Modified Crank / Pin 1 x 3 - Axle Holes"
            },
            {
                "ref": "32054",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 3,
                "name": "Technic, Pin 3L with Friction Ridges and Stop Bush"
            },
            {
                "ref": "32529",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Technic, Pin Connector Plate with Hole on Bottom"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 22,
                "name": "Technic, Pin with Short Friction Ridges"
            },
            {
                "ref": "3069",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Tile 1 x 2"
            },
            {
                "ref": "63864",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Tile 1 x 3"
            },
            {
                "ref": "2431",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 4,
                "name": "Tile 1 x 4"
            },
            {
                "ref": "87079",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Tile 2 x 4"
            },
            {
                "ref": "2412b",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 11,
                "name": "Tile, Modified 1 x 2 Grille with Bottom Groove"
            },
            {
                "ref": "2432",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Tile, Modified 1 x 2 with Bar Handle"
            },
            {
                "ref": "30350b",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 9,
                "name": "Tile, Modified 2 x 3 with 2 Open O Clips"
            },
            {
                "ref": "6180",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Tile, Modified 4 x 6 with Studs on Edges"
            },
            {
                "ref": "3680",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Turntable 2 x 2 Plate, Base"
            },
            {
                "ref": "43710",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Wedge 4 x 2 Triple Left"
            },
            {
                "ref": "43711",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Wedge 4 x 2 Triple Right"
            },
            {
                "ref": "13269",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Wedge 6 x 4 Cutout (Train Roof) with 5 Large Bottom Tubes"
            },
            {
                "ref": "45301",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Wedge 16 x 4 Triple Curved with Reinforcements"
            },
            {
                "ref": "51739",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Wedge, Plate 2 x 4"
            },
            {
                "ref": "41770",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Wedge, Plate 4 x 2 Left"
            },
            {
                "ref": "41769",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 2,
                "name": "Wedge, Plate 4 x 2 Right"
            },
            {
                "ref": "32059",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Wedge, Plate 4 x 6 Cut Corners"
            },
            {
                "ref": "47397",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Wedge, Plate 12 x 3 Left"
            },
            {
                "ref": "47398",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Wedge, Plate 12 x 3 Right"
            },
            {
                "ref": "30355",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Wedge, Plate 12 x 6 Left"
            },
            {
                "ref": "30356",
                "color_code": "11",
                "color_hex": "#202020",
                "color_name": "Black",
                "qty": 1,
                "name": "Wedge, Plate 12 x 6 Right"
            },
            {
                "ref": "43093",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 4,
                "name": "Technic, Axle  1L with Pin with Friction Ridges"
            },
            {
                "ref": "6558",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 15,
                "name": "Technic, Pin 3L with Friction Ridges"
            },
            {
                "ref": "99780",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Bracket 1 x 2 - 1 x 2 Inverted"
            },
            {
                "ref": "44728",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Bracket 1 x 2 - 2 x 2"
            },
            {
                "ref": "93274",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Bracket 1 x 2 - 2 x 4"
            },
            {
                "ref": "4588",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Brick, Round 1 x 1 with Fins"
            },
            {
                "ref": "44567a",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Hinge Plate 1 x 2 Locking with 1 Finger on Side with Bottom Groove"
            },
            {
                "ref": "30383",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Hinge Plate 1 x 2 Locking with 1 Finger on Top"
            },
            {
                "ref": "44300",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Hinge Tile 1 x 3 Locking with 1 Finger on Top"
            },
            {
                "ref": "3666",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Plate 1 x 6"
            },
            {
                "ref": "3020",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 11,
                "name": "Plate 2 x 4"
            },
            {
                "ref": "3795",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 12,
                "name": "Plate 2 x 6"
            },
            {
                "ref": "3839b",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Plate, Modified 1 x 2 with Bar Handles - Flat Ends, Low Attachment"
            },
            {
                "ref": "11458",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 8,
                "name": "Plate, Modified 1 x 2 with Pin Hole on Top"
            },
            {
                "ref": "2444",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Plate, Modified 2 x 2 with Pin Hole"
            },
            {
                "ref": "61409",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Slope 18 2 x 1 x 2/3 with Grille"
            },
            {
                "ref": "3044c",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Slope 45 2 x 1 Double with Bottom Stud Holder"
            },
            {
                "ref": "92946",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Slope 45 2 x 1 with 2/3 Cutout"
            },
            {
                "ref": "3678b",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Slope 65 2 x 2 x 2 with Bottom Tube"
            },
            {
                "ref": "11477",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Slope, Curved 2 x 1 x 2/3"
            },
            {
                "ref": "4185",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Technic Wedge Belt Wheel (Pulley)"
            },
            {
                "ref": "32209",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Technic, Axle  5.5L with Stop"
            },
            {
                "ref": "55013",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Technic, Axle  8L with Stop"
            },
            {
                "ref": "32000",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Technic, Brick 1 x 2 with Holes"
            },
            {
                "ref": "60484",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Technic, Liftarm, Modified T-Shape Thick 3 x 3"
            },
            {
                "ref": "3068",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Tile 2 x 2"
            },
            {
                "ref": "2432",
                "color_code": "85",
                "color_hex": "#6B6D67",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Tile, Modified 1 x 2 with Bar Handle"
            },
            {
                "ref": "6587",
                "color_code": "69",
                "color_hex": "#948972",
                "color_name": "Dark Tan",
                "qty": 2,
                "name": "Technic, Axle  3L with Stud"
            },
            {
                "ref": "30374",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Bar   4L (Lightsaber Blade / Wand)"
            },
            {
                "ref": "87618",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Bar   5L with Handle (Friction Ram)"
            },
            {
                "ref": "2714a",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Bar   8L with Stop Rings and Pin (Technic, Figure Accessory Ski Pole) - Rounded End"
            },
            {
                "ref": "3001",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Brick 2 x 4"
            },
            {
                "ref": "2877",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Brick, Modified 1 x 2 with Grille / Fluted Profile"
            },
            {
                "ref": "98100",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Cone 2 x 2 Truncated"
            },
            {
                "ref": "6233",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Cone 3 x 3 x 2"
            },
            {
                "ref": "60471",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Hinge Plate 1 x 2 Locking with 2 Fingers on Side"
            },
            {
                "ref": "98286",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Hinge Plate 2 x 4 with Pin Hole and 3 Holes - Top"
            },
            {
                "ref": "44570",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Hinge Plate 3 x 4 Locking Dual 2 Fingers"
            },
            {
                "ref": "87544",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Panel 1 x 2 x 3 with Side Supports - Hollow Studs"
            },
            {
                "ref": "3023",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 10,
                "name": "Plate 1 x 2"
            },
            {
                "ref": "3460",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Plate 1 x 8"
            },
            {
                "ref": "3021",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Plate 2 x 3"
            },
            {
                "ref": "3034",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Plate 2 x 8"
            },
            {
                "ref": "32028",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Plate, Modified 1 x 2 with Door Rail"
            },
            {
                "ref": "92593",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Plate, Modified 1 x 4 with 2 Studs without Groove"
            },
            {
                "ref": "87580",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Plate, Modified 2 x 2 with Groove and 1 Stud in Center (Jumper)"
            },
            {
                "ref": "10247",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Plate, Modified 2 x 2 with Pin Hole - Full Cross Support Underneath"
            },
            {
                "ref": "4697b",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Pneumatic T Piece Second Version (T Bar with Ball in Center)"
            },
            {
                "ref": "4265c",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic Bush 1/2 Smooth"
            },
            {
                "ref": "4519",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Axle  3L"
            },
            {
                "ref": "32073",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Technic, Axle  5L"
            },
            {
                "ref": "44294",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Axle  7L"
            },
            {
                "ref": "6536",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Technic, Axle and Pin Connector Perpendicular"
            },
            {
                "ref": "32064",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Brick 1 x 2 with Axle Hole"
            },
            {
                "ref": "3701",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Technic, Brick 1 x 4 with Holes"
            },
            {
                "ref": "60483",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Technic, Liftarm Thick 1 x 2 - Axle Hole"
            },
            {
                "ref": "40490",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Liftarm Thick 1 x 9"
            },
            {
                "ref": "61184",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Pin 1/2 with 2L Bar Extension (Flick Missile)"
            },
            {
                "ref": "4274",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 7,
                "name": "Technic, Pin 1/2 without Friction Ridges"
            },
            {
                "ref": "32557",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Technic, Pin Connector Perpendicular Double 3L"
            },
            {
                "ref": "3673",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Technic, Pin without Friction Ridges"
            },
            {
                "ref": "3068",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Tile 2 x 2"
            },
            {
                "ref": "87079",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Tile 2 x 4"
            },
            {
                "ref": "3679",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Turntable 2 x 2 Plate, Top"
            },
            {
                "ref": "60208",
                "color_code": "86",
                "color_hex": "#A2A1A3",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Wheel 31mm D. x 15mm Technic"
            },
            {
                "ref": "96874",
                "color_code": "4",
                "color_hex": "#CF650F",
                "color_name": "Orange",
                "qty": 1,
                "name": "Brick and Axle Separator"
            },
            {
                "ref": "2436",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 6,
                "name": "Bracket 1 x 2 - 1 x 4"
            },
            {
                "ref": "99207",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Bracket 1 x 2 - 2 x 2 Inverted"
            },
            {
                "ref": "4589b",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 10,
                "name": "Cone 1 x 1 with Top Groove"
            },
            {
                "ref": "3023",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 19,
                "name": "Plate 1 x 2"
            },
            {
                "ref": "3710",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Plate 1 x 4"
            },
            {
                "ref": "3666",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Plate 1 x 6"
            },
            {
                "ref": "3022",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 10,
                "name": "Plate 2 x 2"
            },
            {
                "ref": "2420",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Plate 2 x 2 Corner"
            },
            {
                "ref": "3795",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1,
                "name": "Plate 2 x 6"
            },
            {
                "ref": "4282",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1,
                "name": "Plate 2 x 16"
            },
            {
                "ref": "10247",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 4,
                "name": "Plate, Modified 2 x 2 with Pin Hole - Full Cross Support Underneath"
            },
            {
                "ref": "4073",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1,
                "name": "Plate, Round 1 x 1"
            },
            {
                "ref": "60474",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Plate, Round 4 x 4 with Hole"
            },
            {
                "ref": "3040",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Slope 45 2 x 1"
            },
            {
                "ref": "60481",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Slope 65 2 x 1 x 2"
            },
            {
                "ref": "3713",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Technic Bush"
            },
            {
                "ref": "32062",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1,
                "name": "Technic, Axle  2L Notched"
            },
            {
                "ref": "3700",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Technic, Brick 1 x 2 with Hole"
            },
            {
                "ref": "32523",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Technic, Liftarm Thick 1 x 3"
            },
            {
                "ref": "40490",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1,
                "name": "Technic, Liftarm Thick 1 x 9"
            },
            {
                "ref": "32449",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 2,
                "name": "Technic, Liftarm Thin 1 x 4 - Axle Holes"
            },
            {
                "ref": "62462",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 8,
                "name": "Technic, Pin Connector Round 2L with Slot (Pin Joiner Round)"
            },
            {
                "ref": "2431",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 6,
                "name": "Tile 1 x 4"
            },
            {
                "ref": "41770",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1,
                "name": "Wedge, Plate 4 x 2 Left"
            },
            {
                "ref": "41769",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1,
                "name": "Wedge, Plate 4 x 2 Right"
            },
            {
                "ref": "3003",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 1,
                "name": "Brick 2 x 2"
            },
            {
                "ref": "3749",
                "color_code": "2",
                "color_hex": "#E3CC9D",
                "color_name": "Tan",
                "qty": 2,
                "name": "Technic, Axle  1L with Pin without Friction Ridges with Round Hole"
            },
            {
                "ref": "30372p79",
                "color_code": "13",
                "color_hex": "#625E51",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Windscreen 7 x 4 x 1 2/3 with Locking Dual 2 Fingers with SW Pattern"
            },
            {
                "ref": "3024",
                "color_code": "12",
                "color_hex": "#FEFEFE",
                "color_name": "Trans-Clear",
                "qty": 2,
                "name": "Plate 1 x 1"
            },
            {
                "ref": "3024",
                "color_code": "14",
                "color_hex": "#001F9F",
                "color_name": "Trans-Dark Blue",
                "qty": 1,
                "name": "Plate 1 x 1"
            },
            {
                "ref": "30374",
                "color_code": "15",
                "color_hex": "#AEE9EF",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Bar   4L (Lightsaber Blade / Wand)"
            },
            {
                "ref": "3941",
                "color_code": "15",
                "color_hex": "#AEE9EF",
                "color_name": "Trans-Light Blue",
                "qty": 2,
                "name": "Brick, Round 2 x 2 with Axle Hole"
            },
            {
                "ref": "3960",
                "color_code": "15",
                "color_hex": "#AEE9EF",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Dish 4 x 4 Inverted (Radar) with Solid Stud"
            },
            {
                "ref": "4073",
                "color_code": "15",
                "color_hex": "#AEE9EF",
                "color_name": "Trans-Light Blue",
                "qty": 8,
                "name": "Plate, Round 1 x 1"
            },
            {
                "ref": "98585",
                "color_code": "15",
                "color_hex": "#AEE9EF",
                "color_name": "Trans-Light Blue",
                "qty": 2,
                "name": "Technic, Axle Connector Block Round with 2 Pin Holes and 3 Axle Holes (Hero Factory Weapon Barrel)"
            },
            {
                "ref": "64567",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Minifigure, Weapon Lightsaber Hilt Straight"
            },
            {
                "ref": "2445",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Plate 2 x 12"
            },
            {
                "ref": "4032",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 2,
                "name": "Plate, Round 2 x 2 with Axle Hole"
            }
        ]
    },
    "6008-1": {
        "name": "Royal King",
        "minifigures": [
            {
                "ref": "cas060a",
                "name": "Royal Knights - King, with Blue Legs without Cape and Plume",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "4495a",
                "color_code": "7",
                "color_hex": "#0032B1",
                "color_name": "Blue",
                "qty": 1,
                "name": "Flag 4 x 1 Wave Left"
            },
            {
                "ref": "59",
                "color_code": "22",
                "color_hex": "#DFDFDF",
                "color_name": "Chrome Silver",
                "qty": 1,
                "name": "Minifigure, Weapon Sword, Greatsword Round"
            },
            {
                "ref": "3849",
                "color_code": "10",
                "color_hex": "#6C6D5B",
                "color_name": "Dark Gray",
                "qty": 1,
                "name": "Minifigure, Weapon Lance"
            },
            {
                "ref": "4491b",
                "color_code": "5",
                "color_hex": "#C30025",
                "color_name": "Red",
                "qty": 1,
                "name": "Horse Saddle with Clips"
            },
            {
                "ref": "3004",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Brick 1 x 2"
            },
            {
                "ref": "4493c01pb04",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Horse with Black Eyes, White Pupils and Dark Orange Bridle Pattern"
            },
            {
                "ref": "2586p4d",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Minifigure, Shield Oval with Lion Head, Red and White Background, Blue Border Pattern"
            },
            {
                "ref": "3023",
                "color_code": "1",
                "color_hex": "#F9F9F9",
                "color_name": "White",
                "qty": 1,
                "name": "Plate 1 x 2"
            }
        ]
    },
    "79006-1": {
        "name": "The Council of Elrond",
        "minifigures": [
            {
                "ref": "fig-005192",
                "name": "Gimli",
                "qty": 1
            },
            {
                "ref": "fig-005246",
                "name": "Frodo Baggins - Reddish Brown Torso",
                "qty": 1
            },
            {
                "ref": "fig-005247",
                "name": "Elrond - Dark Red Cape",
                "qty": 1
            },
            {
                "ref": "fig-005248",
                "name": "Arwen",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "10053",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 1,
                "name": "Part 10053"
            },
            {
                "ref": "10053",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 1,
                "name": "Part 10053"
            },
            {
                "ref": "10247",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 10247"
            },
            {
                "ref": "11010",
                "color_code": "21",
                "color_hex": "#BBA53D",
                "color_name": "Chrome Gold",
                "qty": 2,
                "name": "Part 11010"
            },
            {
                "ref": "11010",
                "color_code": "21",
                "color_hex": "#BBA53D",
                "color_name": "Chrome Gold",
                "qty": 1,
                "name": "Part 11010"
            },
            {
                "ref": "11156",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 11156"
            },
            {
                "ref": "11156",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 11156"
            },
            {
                "ref": "13965",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 13965"
            },
            {
                "ref": "14633",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 14633"
            },
            {
                "ref": "2339",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 4,
                "name": "Part 2339"
            },
            {
                "ref": "2417",
                "color_code": "68",
                "color_hex": "#A95500",
                "color_name": "Dark Orange",
                "qty": 3,
                "name": "Part 2417"
            },
            {
                "ref": "2420",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 2420"
            },
            {
                "ref": "2423",
                "color_code": "155",
                "color_hex": "#9B9A5A",
                "color_name": "Olive Green",
                "qty": 5,
                "name": "Part 2423"
            },
            {
                "ref": "2431",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2431"
            },
            {
                "ref": "2445",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2445"
            },
            {
                "ref": "2450",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Part 2450"
            },
            {
                "ref": "2458",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 2458"
            },
            {
                "ref": "2462",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 2462"
            },
            {
                "ref": "2540",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 2540"
            },
            {
                "ref": "2653",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 2653"
            },
            {
                "ref": "2654",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 2654"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 2780"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2780"
            },
            {
                "ref": "2817",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2817"
            },
            {
                "ref": "3004",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 9,
                "name": "Part 3004"
            },
            {
                "ref": "3005",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 3005"
            },
            {
                "ref": "3008",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 3008"
            },
            {
                "ref": "3010",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 3010"
            },
            {
                "ref": "3010",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3010"
            },
            {
                "ref": "30136",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 8,
                "name": "Part 30136"
            },
            {
                "ref": "3020",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3020"
            },
            {
                "ref": "3020",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Part 3020"
            },
            {
                "ref": "3021",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 3021"
            },
            {
                "ref": "3022",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 3022"
            },
            {
                "ref": "3022",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 3023"
            },
            {
                "ref": "30237a",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 30237a"
            },
            {
                "ref": "3030",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 2,
                "name": "Part 3030"
            },
            {
                "ref": "3034",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 3034"
            },
            {
                "ref": "3034",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3034"
            },
            {
                "ref": "30350b",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 3,
                "name": "Part 30350b"
            },
            {
                "ref": "3037",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3037"
            },
            {
                "ref": "30374",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 30374"
            },
            {
                "ref": "3040b",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 3040b"
            },
            {
                "ref": "3040b",
                "color_code": "120",
                "color_hex": "#352100",
                "color_name": "Dark Brown",
                "qty": 6,
                "name": "Part 3040b"
            },
            {
                "ref": "30565",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 4,
                "name": "Part 30565"
            },
            {
                "ref": "3069b",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 9,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3069b"
            },
            {
                "ref": "32028",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 32028"
            },
            {
                "ref": "3460",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 3460"
            },
            {
                "ref": "3622",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3622"
            },
            {
                "ref": "3623",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 7,
                "name": "Part 3623"
            },
            {
                "ref": "3623",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 3623"
            },
            {
                "ref": "3633",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 2,
                "name": "Part 3633"
            },
            {
                "ref": "3660",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 3660"
            },
            {
                "ref": "3665",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3665"
            },
            {
                "ref": "3665",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 3665"
            },
            {
                "ref": "3679",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 3679"
            },
            {
                "ref": "3680",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 3680"
            },
            {
                "ref": "3701",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3701"
            },
            {
                "ref": "3710",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 7,
                "name": "Part 3710"
            },
            {
                "ref": "3795",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3795"
            },
            {
                "ref": "4286",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 4286"
            },
            {
                "ref": "43888",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 43888"
            },
            {
                "ref": "4460b",
                "color_code": "120",
                "color_hex": "#352100",
                "color_name": "Dark Brown",
                "qty": 6,
                "name": "Part 4460b"
            },
            {
                "ref": "48336",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 48336"
            },
            {
                "ref": "4871",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 3,
                "name": "Part 4871"
            },
            {
                "ref": "53454",
                "color_code": "111",
                "color_hex": "#05131D",
                "color_name": "Speckle Black-Silver",
                "qty": 1,
                "name": "Part 53454"
            },
            {
                "ref": "54200",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "155",
                "color_hex": "#9B9A5A",
                "color_name": "Olive Green",
                "qty": 2,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "155",
                "color_hex": "#9B9A5A",
                "color_name": "Olive Green",
                "qty": 4,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 6,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "60474",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 2,
                "name": "Part 60474"
            },
            {
                "ref": "6106",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 6106"
            },
            {
                "ref": "6108",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 6108"
            },
            {
                "ref": "6141",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 7,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6232",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 6232"
            },
            {
                "ref": "6636",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 6636"
            },
            {
                "ref": "85863",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 85863"
            },
            {
                "ref": "87079",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 87079"
            },
            {
                "ref": "87580",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 87580"
            },
            {
                "ref": "87580",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 87580"
            },
            {
                "ref": "91988",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 91988"
            },
            {
                "ref": "92950",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 92950"
            },
            {
                "ref": "93231",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 93231"
            },
            {
                "ref": "93231",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 93231"
            },
            {
                "ref": "93606",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 93606"
            }
        ]
    },
    "75075-1": {
        "name": "AT-AT",
        "minifigures": [
            {
                "ref": "fig-000142",
                "name": "AT-AT Driver, Sand Blue Uniform",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "11458",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 11458"
            },
            {
                "ref": "15712",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 15712"
            },
            {
                "ref": "18677",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 18677"
            },
            {
                "ref": "2357",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2357"
            },
            {
                "ref": "2412b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2412b"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 2780"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2780"
            },
            {
                "ref": "298c02",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 298c02"
            },
            {
                "ref": "298c02",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 298c02"
            },
            {
                "ref": "3005",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3005"
            },
            {
                "ref": "3010",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3010"
            },
            {
                "ref": "3020",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 3020"
            },
            {
                "ref": "3022",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3022"
            },
            {
                "ref": "3022",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 3023"
            },
            {
                "ref": "3039",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3039"
            },
            {
                "ref": "30602",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 30602"
            },
            {
                "ref": "3176",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3176"
            },
            {
                "ref": "32028",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 32028"
            },
            {
                "ref": "3660",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3660"
            },
            {
                "ref": "3839b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3839b"
            },
            {
                "ref": "40902a",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 40902a"
            },
            {
                "ref": "4286",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 4286"
            },
            {
                "ref": "43898",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 43898"
            },
            {
                "ref": "44567a",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 44567a"
            },
            {
                "ref": "4740",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 4740"
            },
            {
                "ref": "48729b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 48729b"
            },
            {
                "ref": "48729b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 48729b"
            },
            {
                "ref": "54200",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "59900",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 2,
                "name": "Part 59900"
            },
            {
                "ref": "60478",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 60478"
            },
            {
                "ref": "61184",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 61184"
            },
            {
                "ref": "6141",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6541",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 6541"
            },
            {
                "ref": "85984",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 85984"
            },
            {
                "ref": "92738",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 92738"
            },
            {
                "ref": "99207",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 99207"
            },
            {
                "ref": "99781",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 99781"
            }
        ]
    },
    "31037-1": {
        "name": "Adventure Vehicles",
        "minifigures": [],
        "parts": [
            {
                "ref": "10928",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 10928"
            },
            {
                "ref": "11477",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 2,
                "name": "Part 11477"
            },
            {
                "ref": "11477",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 11477"
            },
            {
                "ref": "15573",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 15573"
            },
            {
                "ref": "15712",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 15712"
            },
            {
                "ref": "16091",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 16091"
            },
            {
                "ref": "18671",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 18671"
            },
            {
                "ref": "2412b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 5,
                "name": "Part 2412b"
            },
            {
                "ref": "2420",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 2420"
            },
            {
                "ref": "2431",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 2,
                "name": "Part 2431"
            },
            {
                "ref": "2431",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 5,
                "name": "Part 2431"
            },
            {
                "ref": "2432",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 2432"
            },
            {
                "ref": "2584",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2584"
            },
            {
                "ref": "2585",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2585"
            },
            {
                "ref": "2654",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 2654"
            },
            {
                "ref": "2921",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 2921"
            },
            {
                "ref": "3004",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 3004"
            },
            {
                "ref": "3010",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 3010"
            },
            {
                "ref": "3020",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 3020"
            },
            {
                "ref": "3022",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 4,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 3,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 8,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 2,
                "name": "Part 3023"
            },
            {
                "ref": "3024",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 3024"
            },
            {
                "ref": "3032",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3032"
            },
            {
                "ref": "30395",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 30395"
            },
            {
                "ref": "3062b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 11,
                "name": "Part 3062b"
            },
            {
                "ref": "3062b",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 3,
                "name": "Part 3062b"
            },
            {
                "ref": "3068b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3068b"
            },
            {
                "ref": "3069b",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 2,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 3069b"
            },
            {
                "ref": "3070b",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 2,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3176",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3176"
            },
            {
                "ref": "32123b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 32123b"
            },
            {
                "ref": "32123b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 32123b"
            },
            {
                "ref": "32124",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 32124"
            },
            {
                "ref": "32523",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 32523"
            },
            {
                "ref": "32529",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 32529"
            },
            {
                "ref": "3460",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 3460"
            },
            {
                "ref": "3623",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 3623"
            },
            {
                "ref": "3623",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3623"
            },
            {
                "ref": "3660",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3660"
            },
            {
                "ref": "3665",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3665"
            },
            {
                "ref": "3666",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 2,
                "name": "Part 3666"
            },
            {
                "ref": "3666",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 5,
                "name": "Part 3666"
            },
            {
                "ref": "3673",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3673"
            },
            {
                "ref": "3673",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3673"
            },
            {
                "ref": "3700",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3700"
            },
            {
                "ref": "3702",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3702"
            },
            {
                "ref": "3710",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 3710"
            },
            {
                "ref": "3713",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3713"
            },
            {
                "ref": "3713",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3713"
            },
            {
                "ref": "3737",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3737"
            },
            {
                "ref": "3747b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3747b"
            },
            {
                "ref": "3749",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3749"
            },
            {
                "ref": "3795",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 1,
                "name": "Part 3795"
            },
            {
                "ref": "3894",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3894"
            },
            {
                "ref": "3958",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3958"
            },
            {
                "ref": "4032a",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 4032a"
            },
            {
                "ref": "4176",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Part 4176"
            },
            {
                "ref": "4599b",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 4599b"
            },
            {
                "ref": "4599b",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 4599b"
            },
            {
                "ref": "4740",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 2,
                "name": "Part 4740"
            },
            {
                "ref": "47457",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 47457"
            },
            {
                "ref": "48336",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 48336"
            },
            {
                "ref": "50950",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 2,
                "name": "Part 50950"
            },
            {
                "ref": "54200",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 6,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "55982",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 55982"
            },
            {
                "ref": "56823c50",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 56823c50"
            },
            {
                "ref": "56891",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 56891"
            },
            {
                "ref": "59426",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 59426"
            },
            {
                "ref": "6005",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 6,
                "name": "Part 6005"
            },
            {
                "ref": "60470b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 60470b"
            },
            {
                "ref": "60478",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 60478"
            },
            {
                "ref": "61252",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 61252"
            },
            {
                "ref": "61409",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 61409"
            },
            {
                "ref": "6141",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 8,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 9,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "62462",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 62462"
            },
            {
                "ref": "63965",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 63965"
            },
            {
                "ref": "6587",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Part 6587"
            },
            {
                "ref": "85984",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 7,
                "name": "Part 85984"
            },
            {
                "ref": "85984",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 85984"
            },
            {
                "ref": "87079",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 87079"
            },
            {
                "ref": "87087",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 87087"
            },
            {
                "ref": "87609",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 87609"
            },
            {
                "ref": "88072",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 88072"
            },
            {
                "ref": "92280",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 92280"
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 2,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "99206",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 99206"
            },
            {
                "ref": "99780",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 99780"
            }
        ]
    },
    "79012-1": {
        "name": "Mirkwood Elf Army",
        "minifigures": [
            {
                "ref": "fig-005910",
                "name": "Gundabad Orc - Bald and Shoulder Spikes",
                "qty": 1
            },
            {
                "ref": "fig-005911",
                "name": "Gundabad Orc - Bald",
                "qty": 1
            },
            {
                "ref": "fig-005922",
                "name": "Mirkwood Elf  - Dark Green",
                "qty": 1
            },
            {
                "ref": "fig-005923",
                "name": "Mirkwood Elf Archer",
                "qty": 2
            },
            {
                "ref": "fig-005924",
                "name": "Thranduil",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "11429pr0001",
                "color_code": "120",
                "color_hex": "#352100",
                "color_name": "Dark Brown",
                "qty": 1,
                "name": "Part 11429pr0001"
            },
            {
                "ref": "13206pr0003",
                "color_code": "120",
                "color_hex": "#352100",
                "color_name": "Dark Brown",
                "qty": 1,
                "name": "Part 13206pr0003"
            },
            {
                "ref": "13965",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 4,
                "name": "Part 13965"
            },
            {
                "ref": "14413",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 14413"
            },
            {
                "ref": "2339",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 3,
                "name": "Part 2339"
            },
            {
                "ref": "2357",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 2357"
            },
            {
                "ref": "2417",
                "color_code": "288",
                "color_hex": "#184632",
                "color_name": "Dark Green",
                "qty": 1,
                "name": "Part 2417"
            },
            {
                "ref": "2419",
                "color_code": "288",
                "color_hex": "#184632",
                "color_name": "Dark Green",
                "qty": 3,
                "name": "Part 2419"
            },
            {
                "ref": "2420",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 2420"
            },
            {
                "ref": "2423",
                "color_code": "155",
                "color_hex": "#9B9A5A",
                "color_name": "Olive Green",
                "qty": 3,
                "name": "Part 2423"
            },
            {
                "ref": "2449",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 2449"
            },
            {
                "ref": "2450",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 2450"
            },
            {
                "ref": "2489",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 2489"
            },
            {
                "ref": "2586pr0010",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2586pr0010"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 2780"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2780"
            },
            {
                "ref": "2817",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2817"
            },
            {
                "ref": "3003",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 2,
                "name": "Part 3003"
            },
            {
                "ref": "3004",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "120",
                "color_hex": "#352100",
                "color_name": "Dark Brown",
                "qty": 3,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3004"
            },
            {
                "ref": "3005",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 6,
                "name": "Part 3005"
            },
            {
                "ref": "30055",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 30055"
            },
            {
                "ref": "3008",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 3008"
            },
            {
                "ref": "3009",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 3,
                "name": "Part 3009"
            },
            {
                "ref": "3010",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 3010"
            },
            {
                "ref": "30136",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 11,
                "name": "Part 30136"
            },
            {
                "ref": "30137",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 30137"
            },
            {
                "ref": "30153",
                "color_code": "20",
                "color_hex": "#84B68D",
                "color_name": "Trans-Green",
                "qty": 1,
                "name": "Part 30153"
            },
            {
                "ref": "3020",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 3020"
            },
            {
                "ref": "3022",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "120",
                "color_hex": "#352100",
                "color_name": "Dark Brown",
                "qty": 6,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "155",
                "color_hex": "#9B9A5A",
                "color_name": "Olive Green",
                "qty": 4,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 6,
                "name": "Part 3023"
            },
            {
                "ref": "3024",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 3024"
            },
            {
                "ref": "30357",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 6,
                "name": "Part 30357"
            },
            {
                "ref": "3040b",
                "color_code": "120",
                "color_hex": "#352100",
                "color_name": "Dark Brown",
                "qty": 6,
                "name": "Part 3040b"
            },
            {
                "ref": "3045",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3045"
            },
            {
                "ref": "30565",
                "color_code": "288",
                "color_hex": "#184632",
                "color_name": "Dark Green",
                "qty": 5,
                "name": "Part 30565"
            },
            {
                "ref": "3062b",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 9,
                "name": "Part 3062b"
            },
            {
                "ref": "3069b",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 5,
                "name": "Part 3069b"
            },
            {
                "ref": "32000",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 32000"
            },
            {
                "ref": "3245c",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 3245c"
            },
            {
                "ref": "32557",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 32557"
            },
            {
                "ref": "3460",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3460"
            },
            {
                "ref": "3622",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 3622"
            },
            {
                "ref": "3623",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3623"
            },
            {
                "ref": "3623",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3623"
            },
            {
                "ref": "3660",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 3,
                "name": "Part 3660"
            },
            {
                "ref": "3673",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 3673"
            },
            {
                "ref": "3673",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3673"
            },
            {
                "ref": "3700",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 4,
                "name": "Part 3700"
            },
            {
                "ref": "3710",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 3,
                "name": "Part 3710"
            },
            {
                "ref": "3832",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 3832"
            },
            {
                "ref": "3941",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 5,
                "name": "Part 3941"
            },
            {
                "ref": "4032a",
                "color_code": "68",
                "color_hex": "#A95500",
                "color_name": "Dark Orange",
                "qty": 1,
                "name": "Part 4032a"
            },
            {
                "ref": "4070",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 4070"
            },
            {
                "ref": "4274",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 4274"
            },
            {
                "ref": "4274",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 4274"
            },
            {
                "ref": "4286",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 4286"
            },
            {
                "ref": "4491b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 4491b"
            },
            {
                "ref": "4865b",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Part 4865b"
            },
            {
                "ref": "4865b",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 4865b"
            },
            {
                "ref": "54200",
                "color_code": "155",
                "color_hex": "#9B9A5A",
                "color_name": "Olive Green",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "155",
                "color_hex": "#9B9A5A",
                "color_name": "Olive Green",
                "qty": 3,
                "name": "Part 54200"
            },
            {
                "ref": "55236",
                "color_code": "288",
                "color_hex": "#184632",
                "color_name": "Dark Green",
                "qty": 1,
                "name": "Part 55236"
            },
            {
                "ref": "59900",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 59900"
            },
            {
                "ref": "60481",
                "color_code": "120",
                "color_hex": "#352100",
                "color_name": "Dark Brown",
                "qty": 7,
                "name": "Part 60481"
            },
            {
                "ref": "60481",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 60481"
            },
            {
                "ref": "60752",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 1,
                "name": "Part 60752"
            },
            {
                "ref": "60897",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 60897"
            },
            {
                "ref": "61184",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 61184"
            },
            {
                "ref": "61252",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 61252"
            },
            {
                "ref": "6141",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "20",
                "color_hex": "#84B68D",
                "color_name": "Trans-Green",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 9,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "20",
                "color_hex": "#84B68D",
                "color_name": "Trans-Green",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "61485",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 61485"
            },
            {
                "ref": "6231",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 10,
                "name": "Part 6231"
            },
            {
                "ref": "63864",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 4,
                "name": "Part 63864"
            },
            {
                "ref": "6541",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 6541"
            },
            {
                "ref": "6587",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Part 6587"
            },
            {
                "ref": "73983",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 73983"
            },
            {
                "ref": "85863",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 85863"
            },
            {
                "ref": "87079",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 87079"
            },
            {
                "ref": "87081",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 87081"
            },
            {
                "ref": "87994",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 87994"
            },
            {
                "ref": "88288pat0001",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 88288pat0001"
            },
            {
                "ref": "88288pat0001",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 88288pat0001"
            },
            {
                "ref": "88292",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 88292"
            },
            {
                "ref": "89523",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 89523"
            },
            {
                "ref": "92220",
                "color_code": "2",
                "color_hex": "#3E3C39",
                "color_name": "Pearl Titanium",
                "qty": 2,
                "name": "Part 92220"
            },
            {
                "ref": "93231",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 3,
                "name": "Part 93231"
            },
            {
                "ref": "93273",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 93273"
            },
            {
                "ref": "93789",
                "color_code": "2",
                "color_hex": "#3E3C39",
                "color_name": "Pearl Titanium",
                "qty": 1,
                "name": "Part 93789"
            },
            {
                "ref": "98370",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 1,
                "name": "Part 98370"
            },
            {
                "ref": "98560",
                "color_code": "120",
                "color_hex": "#352100",
                "color_name": "Dark Brown",
                "qty": 1,
                "name": "Part 98560"
            }
        ]
    },
    "60008-1": {
        "name": "Museum Break-in",
        "minifigures": [
            {
                "ref": "fig-001277",
                "name": "Policeman, Dark Blue Jacket with Pockets and Badge, White Helmet with Visor",
                "qty": 1
            },
            {
                "ref": "fig-007837",
                "name": "Criminal, Black Striped Top with Rope, Dark Bluish Gray Hat, Stubble",
                "qty": 1
            },
            {
                "ref": "fig-007838",
                "name": "Policeman, Dark Bluish Gray Vest with Radio, Badge and Pouches over Medium Blue Shirt, Black Cap, Beard",
                "qty": 1
            },
            {
                "ref": "fig-007846",
                "name": "Policeman, Dark Bluish Gray Vest with Radio, Badge and Pouches over Medium Blue Shirt, Black Hat with Visor",
                "qty": 1
            },
            {
                "ref": "fig-007847",
                "name": "Policeman, Dark Bluish Gray Vest with Radio, Badge and Pouches over Medium Blue Shirt, Black Hat with Visor, Open Mouth",
                "qty": 1
            },
            {
                "ref": "fig-007848",
                "name": "Criminal, Black Striped Top with Rope, Black Hair, Black Eyemask",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "10201",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 10201"
            },
            {
                "ref": "11211",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 11211"
            },
            {
                "ref": "11289",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 11289"
            },
            {
                "ref": "12752",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 12752"
            },
            {
                "ref": "12825",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 12825"
            },
            {
                "ref": "13561",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 13561"
            },
            {
                "ref": "14210",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 14210"
            },
            {
                "ref": "15207",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 15207"
            },
            {
                "ref": "2412b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 5,
                "name": "Part 2412b"
            },
            {
                "ref": "2419",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 2419"
            },
            {
                "ref": "2421",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2421"
            },
            {
                "ref": "2421",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2421"
            },
            {
                "ref": "2423",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 2,
                "name": "Part 2423"
            },
            {
                "ref": "2431",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2431"
            },
            {
                "ref": "2432",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 2432"
            },
            {
                "ref": "2432",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 2432"
            },
            {
                "ref": "2445",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 2445"
            },
            {
                "ref": "2445",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2445"
            },
            {
                "ref": "2447",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 2,
                "name": "Part 2447"
            },
            {
                "ref": "2456",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 2456"
            },
            {
                "ref": "2456",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 2456"
            },
            {
                "ref": "2465",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 2465"
            },
            {
                "ref": "2479",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2479"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2780"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2780"
            },
            {
                "ref": "298c02",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 298c02"
            },
            {
                "ref": "298c02",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 298c02"
            },
            {
                "ref": "3001",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3001"
            },
            {
                "ref": "3003",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 2,
                "name": "Part 3003"
            },
            {
                "ref": "3003",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3003"
            },
            {
                "ref": "3004",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 8,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 3004"
            },
            {
                "ref": "3008",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 3008"
            },
            {
                "ref": "3008",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 3008"
            },
            {
                "ref": "3009",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3009"
            },
            {
                "ref": "3009",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 6,
                "name": "Part 3009"
            },
            {
                "ref": "3010",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 3010"
            },
            {
                "ref": "3010",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 7,
                "name": "Part 3010"
            },
            {
                "ref": "3010",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3010"
            },
            {
                "ref": "30136",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 16,
                "name": "Part 30136"
            },
            {
                "ref": "30151a",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 30151a"
            },
            {
                "ref": "30153",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 1,
                "name": "Part 30153"
            },
            {
                "ref": "30173b",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 30173b"
            },
            {
                "ref": "30173b",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 30173b"
            },
            {
                "ref": "3020",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 3,
                "name": "Part 3020"
            },
            {
                "ref": "3021",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 3021"
            },
            {
                "ref": "3021",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3021"
            },
            {
                "ref": "3022",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 3,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 4,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 16,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "19",
                "color_hex": "#F5CD2F",
                "color_name": "Trans-Yellow",
                "qty": 2,
                "name": "Part 3023"
            },
            {
                "ref": "3024",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 4,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 2,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "19",
                "color_hex": "#F5CD2F",
                "color_name": "Trans-Yellow",
                "qty": 2,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "19",
                "color_hex": "#F5CD2F",
                "color_name": "Trans-Yellow",
                "qty": 1,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3024"
            },
            {
                "ref": "30248",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 30248"
            },
            {
                "ref": "30292",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 2,
                "name": "Part 30292"
            },
            {
                "ref": "3031",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 3031"
            },
            {
                "ref": "3032",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3032"
            },
            {
                "ref": "3032",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3032"
            },
            {
                "ref": "3034",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 5,
                "name": "Part 3034"
            },
            {
                "ref": "30350c",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 30350c"
            },
            {
                "ref": "30363",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 12,
                "name": "Part 30363"
            },
            {
                "ref": "30374",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 6,
                "name": "Part 30374"
            },
            {
                "ref": "30374",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 6,
                "name": "Part 30374"
            },
            {
                "ref": "30385",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 30385"
            },
            {
                "ref": "3039",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 4,
                "name": "Part 3039"
            },
            {
                "ref": "3040b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3040b"
            },
            {
                "ref": "3040b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3040b"
            },
            {
                "ref": "30414",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 30414"
            },
            {
                "ref": "30592",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 30592"
            },
            {
                "ref": "3062b",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 4,
                "name": "Part 3062b"
            },
            {
                "ref": "3069b",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 2,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "19",
                "color_hex": "#F5CD2F",
                "color_name": "Trans-Yellow",
                "qty": 2,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 15,
                "name": "Part 3069b"
            },
            {
                "ref": "3070b",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "20",
                "color_hex": "#84B68D",
                "color_name": "Trans-Green",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "20",
                "color_hex": "#84B68D",
                "color_name": "Trans-Green",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "32000",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 32000"
            },
            {
                "ref": "32062",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 32062"
            },
            {
                "ref": "3460",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3460"
            },
            {
                "ref": "3622",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 3622"
            },
            {
                "ref": "3623",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3623"
            },
            {
                "ref": "3660",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 2,
                "name": "Part 3660"
            },
            {
                "ref": "3660",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3660"
            },
            {
                "ref": "3665",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 6,
                "name": "Part 3665"
            },
            {
                "ref": "3666",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 9,
                "name": "Part 3666"
            },
            {
                "ref": "3666",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 7,
                "name": "Part 3666"
            },
            {
                "ref": "3700",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3700"
            },
            {
                "ref": "3710",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 5,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3710"
            },
            {
                "ref": "3747b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 3747b"
            },
            {
                "ref": "3794b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3794b"
            },
            {
                "ref": "3794b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3794b"
            },
            {
                "ref": "3795",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3795"
            },
            {
                "ref": "3795",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 3795"
            },
            {
                "ref": "3795",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3795"
            },
            {
                "ref": "3795",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 3795"
            },
            {
                "ref": "3829c01",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 3829c01"
            },
            {
                "ref": "3941",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3941"
            },
            {
                "ref": "3941",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 22,
                "name": "Part 3941"
            },
            {
                "ref": "3958",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3958"
            },
            {
                "ref": "3962b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3962b"
            },
            {
                "ref": "40490",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 40490"
            },
            {
                "ref": "4079b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 4079b"
            },
            {
                "ref": "4079b",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 4079b"
            },
            {
                "ref": "4150pr0001",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 4150pr0001"
            },
            {
                "ref": "4162",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 4162"
            },
            {
                "ref": "4176",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 4176"
            },
            {
                "ref": "4274",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 6,
                "name": "Part 4274"
            },
            {
                "ref": "4274",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 4274"
            },
            {
                "ref": "4349",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 4349"
            },
            {
                "ref": "43713",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 43713"
            },
            {
                "ref": "43719",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 43719"
            },
            {
                "ref": "44567a",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 44567a"
            },
            {
                "ref": "44661",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 44661"
            },
            {
                "ref": "44728",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 4,
                "name": "Part 44728"
            },
            {
                "ref": "4477",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 4477"
            },
            {
                "ref": "4488",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 9,
                "name": "Part 4488"
            },
            {
                "ref": "4740",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 4740"
            },
            {
                "ref": "47457",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 47457"
            },
            {
                "ref": "47753",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 47753"
            },
            {
                "ref": "47998",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 47998"
            },
            {
                "ref": "48092",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 48092"
            },
            {
                "ref": "48336",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 10,
                "name": "Part 48336"
            },
            {
                "ref": "48336",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 2,
                "name": "Part 48336"
            },
            {
                "ref": "48336",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 48336"
            },
            {
                "ref": "4865a",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 4865a"
            },
            {
                "ref": "4865a",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 4865a"
            },
            {
                "ref": "50745",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 50745"
            },
            {
                "ref": "50745",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 50745"
            },
            {
                "ref": "50950",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 50950"
            },
            {
                "ref": "52031",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 52031"
            },
            {
                "ref": "52031",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 52031"
            },
            {
                "ref": "52037",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 52037"
            },
            {
                "ref": "52107",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 52107"
            },
            {
                "ref": "57895",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 2,
                "name": "Part 57895"
            },
            {
                "ref": "6003",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 6003"
            },
            {
                "ref": "6014b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 6014b"
            },
            {
                "ref": "60470b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 60470b"
            },
            {
                "ref": "60476",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 60476"
            },
            {
                "ref": "60478",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 60478"
            },
            {
                "ref": "60581",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 3,
                "name": "Part 60581"
            },
            {
                "ref": "60581",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Part 60581"
            },
            {
                "ref": "60581",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 60581"
            },
            {
                "ref": "60583b",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 6,
                "name": "Part 60583b"
            },
            {
                "ref": "60596",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 60596"
            },
            {
                "ref": "60616a",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 60616a"
            },
            {
                "ref": "6091",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 6091"
            },
            {
                "ref": "6112",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 6112"
            },
            {
                "ref": "6141",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "19",
                "color_hex": "#F5CD2F",
                "color_name": "Trans-Yellow",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "19",
                "color_hex": "#F5CD2F",
                "color_name": "Trans-Yellow",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "61482",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 61482"
            },
            {
                "ref": "61482",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 61482"
            },
            {
                "ref": "6179",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Part 6179"
            },
            {
                "ref": "62113",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 62113"
            },
            {
                "ref": "6256",
                "color_code": "65",
                "color_hex": "#DBAC34",
                "color_name": "Metallic Gold",
                "qty": 1,
                "name": "Part 6256"
            },
            {
                "ref": "62930",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 62930"
            },
            {
                "ref": "63864",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 63864"
            },
            {
                "ref": "64567",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 64567"
            },
            {
                "ref": "64567",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 64567"
            },
            {
                "ref": "6636",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 6636"
            },
            {
                "ref": "6636",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 6636"
            },
            {
                "ref": "72454",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 72454"
            },
            {
                "ref": "85863",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 85863"
            },
            {
                "ref": "85984",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 85984"
            },
            {
                "ref": "85984",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 85984"
            },
            {
                "ref": "87079",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 87079"
            },
            {
                "ref": "87079",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 87079"
            },
            {
                "ref": "87079",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 3,
                "name": "Part 87079"
            },
            {
                "ref": "87079",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 87079"
            },
            {
                "ref": "87087",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 87087"
            },
            {
                "ref": "87552",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 6,
                "name": "Part 87552"
            },
            {
                "ref": "87580",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 87580"
            },
            {
                "ref": "87609",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 87609"
            },
            {
                "ref": "87697",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 8,
                "name": "Part 87697"
            },
            {
                "ref": "91405",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 91405"
            },
            {
                "ref": "92099",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 92099"
            },
            {
                "ref": "92280",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 92280"
            },
            {
                "ref": "92583",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Part 92583"
            },
            {
                "ref": "92585",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 92585"
            },
            {
                "ref": "92593",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 92593"
            },
            {
                "ref": "92593",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 92593"
            },
            {
                "ref": "96874",
                "color_code": "4",
                "color_hex": "#FE8A18",
                "color_name": "Orange",
                "qty": 1,
                "name": "Part 96874"
            },
            {
                "ref": "98138",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 2,
                "name": "Part 98138"
            },
            {
                "ref": "98835",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 98835"
            }
        ]
    },
    "42075-1": {
        "name": "First Responder",
        "minifigures": [],
        "parts": [
            {
                "ref": "10197",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 10197"
            },
            {
                "ref": "11214",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 15,
                "name": "Part 11214"
            },
            {
                "ref": "11946",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 11946"
            },
            {
                "ref": "11947",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 11947"
            },
            {
                "ref": "11954",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 11954"
            },
            {
                "ref": "14720",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 14720"
            },
            {
                "ref": "15100",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 15100"
            },
            {
                "ref": "15413",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 15413"
            },
            {
                "ref": "15458",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 15458"
            },
            {
                "ref": "15461",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 15461"
            },
            {
                "ref": "18575",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 18575"
            },
            {
                "ref": "18651",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 18651"
            },
            {
                "ref": "18654",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 12,
                "name": "Part 18654"
            },
            {
                "ref": "18654",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 18654"
            },
            {
                "ref": "18946",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 18946"
            },
            {
                "ref": "22961",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 22961"
            },
            {
                "ref": "24116",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 24116"
            },
            {
                "ref": "25214",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 25214"
            },
            {
                "ref": "26287",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 5,
                "name": "Part 26287"
            },
            {
                "ref": "2654",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 2654"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 2780"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 77,
                "name": "Part 2780"
            },
            {
                "ref": "27940",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 27940"
            },
            {
                "ref": "2819",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2819"
            },
            {
                "ref": "2850b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2850b"
            },
            {
                "ref": "2851",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 2851"
            },
            {
                "ref": "2852",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2852"
            },
            {
                "ref": "2853",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 2853"
            },
            {
                "ref": "3023",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 2,
                "name": "Part 3023"
            },
            {
                "ref": "30395",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 30395"
            },
            {
                "ref": "3069b",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 2,
                "name": "Part 3069b"
            },
            {
                "ref": "32013",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 32013"
            },
            {
                "ref": "32013",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 3,
                "name": "Part 32013"
            },
            {
                "ref": "32015",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 32015"
            },
            {
                "ref": "32016",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 32016"
            },
            {
                "ref": "32034",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 8,
                "name": "Part 32034"
            },
            {
                "ref": "32039",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 32039"
            },
            {
                "ref": "32054",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 7,
                "name": "Part 32054"
            },
            {
                "ref": "32062",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 14,
                "name": "Part 32062"
            },
            {
                "ref": "32073",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Part 32073"
            },
            {
                "ref": "32140",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 32140"
            },
            {
                "ref": "32184",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Part 32184"
            },
            {
                "ref": "32270",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 6,
                "name": "Part 32270"
            },
            {
                "ref": "32278",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 32278"
            },
            {
                "ref": "32316",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 32316"
            },
            {
                "ref": "32523",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 7,
                "name": "Part 32523"
            },
            {
                "ref": "32524",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 5,
                "name": "Part 32524"
            },
            {
                "ref": "32524",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 32524"
            },
            {
                "ref": "32524",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 32524"
            },
            {
                "ref": "32525",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 32525"
            },
            {
                "ref": "32526",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 5,
                "name": "Part 32526"
            },
            {
                "ref": "32526",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 32526"
            },
            {
                "ref": "32556",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 32556"
            },
            {
                "ref": "3705",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 6,
                "name": "Part 3705"
            },
            {
                "ref": "3706",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 3706"
            },
            {
                "ref": "3710",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3710"
            },
            {
                "ref": "3713",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 17,
                "name": "Part 3713"
            },
            {
                "ref": "3713",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 3713"
            },
            {
                "ref": "3737",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3737"
            },
            {
                "ref": "37609",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 37609"
            },
            {
                "ref": "3835",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3835"
            },
            {
                "ref": "3942c",
                "color_code": "4",
                "color_hex": "#FE8A18",
                "color_name": "Orange",
                "qty": 2,
                "name": "Part 3942c"
            },
            {
                "ref": "4032a",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 4032a"
            },
            {
                "ref": "40490",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 40490"
            },
            {
                "ref": "41239",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 41239"
            },
            {
                "ref": "41678",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 41678"
            },
            {
                "ref": "42003",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 42003"
            },
            {
                "ref": "4274",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4274"
            },
            {
                "ref": "4274",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 4274"
            },
            {
                "ref": "4274",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 15,
                "name": "Part 4274"
            },
            {
                "ref": "4274",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 10,
                "name": "Part 4274"
            },
            {
                "ref": "43093",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 9,
                "name": "Part 43093"
            },
            {
                "ref": "43857",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 43857"
            },
            {
                "ref": "44294",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 44294"
            },
            {
                "ref": "4519",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 15,
                "name": "Part 4519"
            },
            {
                "ref": "4599b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4599b"
            },
            {
                "ref": "4599b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4599b"
            },
            {
                "ref": "48989",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 48989"
            },
            {
                "ref": "54200",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 4,
                "name": "Part 54200"
            },
            {
                "ref": "55013",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 55013"
            },
            {
                "ref": "55615",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 55615"
            },
            {
                "ref": "56145",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 56145"
            },
            {
                "ref": "56823c50",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 56823c50"
            },
            {
                "ref": "59426",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 59426"
            },
            {
                "ref": "59443",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 4,
                "name": "Part 59443"
            },
            {
                "ref": "60483",
                "color_code": "4",
                "color_hex": "#FE8A18",
                "color_name": "Orange",
                "qty": 1,
                "name": "Part 60483"
            },
            {
                "ref": "60484",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 6,
                "name": "Part 60484"
            },
            {
                "ref": "6141",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 6141"
            },
            {
                "ref": "61510",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 61510"
            },
            {
                "ref": "61903",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 61903"
            },
            {
                "ref": "62462",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 62462"
            },
            {
                "ref": "62821b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 62821b"
            },
            {
                "ref": "63869",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 63869"
            },
            {
                "ref": "64178",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 64178"
            },
            {
                "ref": "64179",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 64179"
            },
            {
                "ref": "64391",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 64391"
            },
            {
                "ref": "64683",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 64683"
            },
            {
                "ref": "6536",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 22,
                "name": "Part 6536"
            },
            {
                "ref": "6558",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 40,
                "name": "Part 6558"
            },
            {
                "ref": "6587",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 2,
                "name": "Part 6587"
            },
            {
                "ref": "6589",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 6589"
            },
            {
                "ref": "6629",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 6629"
            },
            {
                "ref": "76537",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 76537"
            },
            {
                "ref": "87082",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 87082"
            },
            {
                "ref": "87083",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 87083"
            },
            {
                "ref": "87408",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 87408"
            },
            {
                "ref": "87761",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 87761"
            },
            {
                "ref": "92907",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 92907"
            },
            {
                "ref": "94925",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 94925"
            },
            {
                "ref": "98138",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 4,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 4,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 6,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 2,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 98138"
            }
        ]
    },
    "75052-1": {
        "name": "Mos Eisley Cantina",
        "minifigures": [
            {
                "ref": "fig-000144",
                "name": "Luke Skywalker, White Robe, White Legs, Cheek Lines",
                "qty": 1
            },
            {
                "ref": "fig-000509",
                "name": "Han Solo, Black Vest, Dark Blue Legs, Smooth Hair, Cheek Lines",
                "qty": 1
            },
            {
                "ref": "fig-002523",
                "name": "Obi-Wan Kenobi, Old, Long Dark Brown Robe",
                "qty": 1
            },
            {
                "ref": "fig-004065",
                "name": "Bith Musician",
                "qty": 3
            },
            {
                "ref": "fig-004066",
                "name": "Greedo, Sand Green Skin, No Knee Strap",
                "qty": 1
            },
            {
                "ref": "fig-004067",
                "name": "Sandtrooper, Black Pauldron, Neck Bracket with Single Stud",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "10247",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 10247"
            },
            {
                "ref": "11153",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 17,
                "name": "Part 11153"
            },
            {
                "ref": "11211",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 11211"
            },
            {
                "ref": "11477",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 8,
                "name": "Part 11477"
            },
            {
                "ref": "12825",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 12825"
            },
            {
                "ref": "12825",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 12825"
            },
            {
                "ref": "12825",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 12825"
            },
            {
                "ref": "14395",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 14395"
            },
            {
                "ref": "15254",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 15254"
            },
            {
                "ref": "15535",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 15535"
            },
            {
                "ref": "16873pr0001",
                "color_code": "155",
                "color_hex": "#9B9A5A",
                "color_name": "Olive Green",
                "qty": 1,
                "name": "Part 16873pr0001"
            },
            {
                "ref": "16875pr0001",
                "color_code": "155",
                "color_hex": "#9B9A5A",
                "color_name": "Olive Green",
                "qty": 1,
                "name": "Part 16875pr0001"
            },
            {
                "ref": "16877",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 16877"
            },
            {
                "ref": "17594",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 17594"
            },
            {
                "ref": "2357",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 8,
                "name": "Part 2357"
            },
            {
                "ref": "2357",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2357"
            },
            {
                "ref": "2412b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 2412b"
            },
            {
                "ref": "2412b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 2412b"
            },
            {
                "ref": "2420",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 2420"
            },
            {
                "ref": "2420",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 4,
                "name": "Part 2420"
            },
            {
                "ref": "2431",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 2431"
            },
            {
                "ref": "2431",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 3,
                "name": "Part 2431"
            },
            {
                "ref": "2460",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2460"
            },
            {
                "ref": "2489",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 2489"
            },
            {
                "ref": "2540",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2540"
            },
            {
                "ref": "2569",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2569"
            },
            {
                "ref": "2569",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 2569"
            },
            {
                "ref": "2653",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 2653"
            },
            {
                "ref": "2654",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 10,
                "name": "Part 2654"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2780"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2780"
            },
            {
                "ref": "2819",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2819"
            },
            {
                "ref": "2877",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 20,
                "name": "Part 2877"
            },
            {
                "ref": "2877",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 2877"
            },
            {
                "ref": "298c02",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 298c02"
            },
            {
                "ref": "298c02",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 298c02"
            },
            {
                "ref": "3001",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 2,
                "name": "Part 3001"
            },
            {
                "ref": "3001",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3001"
            },
            {
                "ref": "3001",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3001"
            },
            {
                "ref": "3003",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 3003"
            },
            {
                "ref": "3003",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 7,
                "name": "Part 3003"
            },
            {
                "ref": "3004",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 3,
                "name": "Part 3004"
            },
            {
                "ref": "3005",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3005"
            },
            {
                "ref": "3005",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3005"
            },
            {
                "ref": "3005",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 21,
                "name": "Part 3005"
            },
            {
                "ref": "3005",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 3005"
            },
            {
                "ref": "3008",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3008"
            },
            {
                "ref": "3009",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 5,
                "name": "Part 3009"
            },
            {
                "ref": "3010",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3010"
            },
            {
                "ref": "3010",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 3,
                "name": "Part 3010"
            },
            {
                "ref": "3010",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 3010"
            },
            {
                "ref": "3020",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 3,
                "name": "Part 3020"
            },
            {
                "ref": "3020",
                "color_code": "28",
                "color_hex": "#D09168",
                "color_name": "Nougat",
                "qty": 5,
                "name": "Part 3020"
            },
            {
                "ref": "3020",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 1,
                "name": "Part 3020"
            },
            {
                "ref": "3022",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3022"
            },
            {
                "ref": "3022",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 5,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 24,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 24,
                "name": "Part 3023"
            },
            {
                "ref": "30237a",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 9,
                "name": "Part 30237a"
            },
            {
                "ref": "3024",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 1,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 1,
                "name": "Part 3024"
            },
            {
                "ref": "30304",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 30304"
            },
            {
                "ref": "3031",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 3031"
            },
            {
                "ref": "3032",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3032"
            },
            {
                "ref": "3034",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 3,
                "name": "Part 3034"
            },
            {
                "ref": "3035",
                "color_code": "28",
                "color_hex": "#D09168",
                "color_name": "Nougat",
                "qty": 4,
                "name": "Part 3035"
            },
            {
                "ref": "3035",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 4,
                "name": "Part 3035"
            },
            {
                "ref": "30374",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 2,
                "name": "Part 30374"
            },
            {
                "ref": "30374",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 2,
                "name": "Part 30374"
            },
            {
                "ref": "3039",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3039"
            },
            {
                "ref": "3039",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3039"
            },
            {
                "ref": "3040b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3040b"
            },
            {
                "ref": "3040b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3040b"
            },
            {
                "ref": "3040b",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 6,
                "name": "Part 3040b"
            },
            {
                "ref": "3040b",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 5,
                "name": "Part 3040b"
            },
            {
                "ref": "30562",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 30562"
            },
            {
                "ref": "30565",
                "color_code": "28",
                "color_hex": "#D09168",
                "color_name": "Nougat",
                "qty": 4,
                "name": "Part 30565"
            },
            {
                "ref": "30586",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 30586"
            },
            {
                "ref": "3062b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3062b"
            },
            {
                "ref": "3062b",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 4,
                "name": "Part 3062b"
            },
            {
                "ref": "3062b",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 2,
                "name": "Part 3062b"
            },
            {
                "ref": "3062b",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 3,
                "name": "Part 3062b"
            },
            {
                "ref": "3062b",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 3062b"
            },
            {
                "ref": "3068b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3068b"
            },
            {
                "ref": "3068b",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 3068b"
            },
            {
                "ref": "3069b",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 2,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 11,
                "name": "Part 3069b"
            },
            {
                "ref": "3070b",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "32013",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 6,
                "name": "Part 32013"
            },
            {
                "ref": "32028",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 2,
                "name": "Part 32028"
            },
            {
                "ref": "32034",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 32034"
            },
            {
                "ref": "32062",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 32062"
            },
            {
                "ref": "32064a",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 32064a"
            },
            {
                "ref": "32530",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 32530"
            },
            {
                "ref": "3622",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 3622"
            },
            {
                "ref": "3622",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3622"
            },
            {
                "ref": "3623",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3623"
            },
            {
                "ref": "3665",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3665"
            },
            {
                "ref": "3666",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Part 3666"
            },
            {
                "ref": "3700",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3700"
            },
            {
                "ref": "3700",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3700"
            },
            {
                "ref": "3706",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3706"
            },
            {
                "ref": "3710",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 8,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 9,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3710"
            },
            {
                "ref": "3794b",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 3794b"
            },
            {
                "ref": "3794b",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 3,
                "name": "Part 3794b"
            },
            {
                "ref": "3795",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 3795"
            },
            {
                "ref": "3795",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3795"
            },
            {
                "ref": "3795",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3795"
            },
            {
                "ref": "3829c01",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3829c01"
            },
            {
                "ref": "3899",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 3899"
            },
            {
                "ref": "3941",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3941"
            },
            {
                "ref": "4032a",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 4032a"
            },
            {
                "ref": "4079b",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 4079b"
            },
            {
                "ref": "4079b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 4079b"
            },
            {
                "ref": "41539",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 2,
                "name": "Part 41539"
            },
            {
                "ref": "4162",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 4162"
            },
            {
                "ref": "4175",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Part 4175"
            },
            {
                "ref": "42446",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 42446"
            },
            {
                "ref": "42610",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 42610"
            },
            {
                "ref": "4274",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4274"
            },
            {
                "ref": "4274",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 7,
                "name": "Part 4274"
            },
            {
                "ref": "43337",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 43337"
            },
            {
                "ref": "44728",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 44728"
            },
            {
                "ref": "4477",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 4477"
            },
            {
                "ref": "4477",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 4477"
            },
            {
                "ref": "4519",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4519"
            },
            {
                "ref": "4536",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 4536"
            },
            {
                "ref": "4599b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 4599b"
            },
            {
                "ref": "4599b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 4599b"
            },
            {
                "ref": "48092",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 48092"
            },
            {
                "ref": "4865b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 4865b"
            },
            {
                "ref": "4865b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 9,
                "name": "Part 4865b"
            },
            {
                "ref": "4868b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 4868b"
            },
            {
                "ref": "4868b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4868b"
            },
            {
                "ref": "50950",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 1,
                "name": "Part 50950"
            },
            {
                "ref": "54200",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 4,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 4,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "55013",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 55013"
            },
            {
                "ref": "57585",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 57585"
            },
            {
                "ref": "57899",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 57899"
            },
            {
                "ref": "59900",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 59900"
            },
            {
                "ref": "59900",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 59900"
            },
            {
                "ref": "59900",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 59900"
            },
            {
                "ref": "59900",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 59900"
            },
            {
                "ref": "60475b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 60475b"
            },
            {
                "ref": "60478",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 60478"
            },
            {
                "ref": "60808",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 5,
                "name": "Part 60808"
            },
            {
                "ref": "60897",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 60897"
            },
            {
                "ref": "6091",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 6091"
            },
            {
                "ref": "61252",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 7,
                "name": "Part 61252"
            },
            {
                "ref": "61252",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 4,
                "name": "Part 61252"
            },
            {
                "ref": "61409",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 61409"
            },
            {
                "ref": "6141",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 6,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "62360",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 62360"
            },
            {
                "ref": "63965",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 63965"
            },
            {
                "ref": "64567",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 64567"
            },
            {
                "ref": "64567",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 64567"
            },
            {
                "ref": "64567",
                "color_code": "67",
                "color_hex": "#A5A9B4",
                "color_name": "Metallic Silver",
                "qty": 2,
                "name": "Part 64567"
            },
            {
                "ref": "64567",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 64567"
            },
            {
                "ref": "64567",
                "color_code": "67",
                "color_hex": "#A5A9B4",
                "color_name": "Metallic Silver",
                "qty": 2,
                "name": "Part 64567"
            },
            {
                "ref": "64567",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 64567"
            },
            {
                "ref": "64644",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 64644"
            },
            {
                "ref": "73983",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 7,
                "name": "Part 73983"
            },
            {
                "ref": "75c12",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 1,
                "name": "Part 75c12"
            },
            {
                "ref": "75c20",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 75c20"
            },
            {
                "ref": "75c20",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 75c20"
            },
            {
                "ref": "85080",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 85080"
            },
            {
                "ref": "85984",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 85984"
            },
            {
                "ref": "87087",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 6,
                "name": "Part 87087"
            },
            {
                "ref": "87087",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 87087"
            },
            {
                "ref": "87087",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 87087"
            },
            {
                "ref": "87544",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 87544"
            },
            {
                "ref": "87552",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 87552"
            },
            {
                "ref": "87580",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 87580"
            },
            {
                "ref": "87618",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 87618"
            },
            {
                "ref": "88072",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 88072"
            },
            {
                "ref": "91501",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 91501"
            },
            {
                "ref": "92410",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 92410"
            },
            {
                "ref": "92582",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 92582"
            },
            {
                "ref": "92593",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 92593"
            },
            {
                "ref": "92690",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 5,
                "name": "Part 92690"
            },
            {
                "ref": "92690",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 92690"
            },
            {
                "ref": "92691",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 92691"
            },
            {
                "ref": "92738",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 92738"
            },
            {
                "ref": "92947",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 92947"
            },
            {
                "ref": "93273",
                "color_code": "155",
                "color_hex": "#9B9A5A",
                "color_name": "Olive Green",
                "qty": 4,
                "name": "Part 93273"
            },
            {
                "ref": "95198",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 95198"
            },
            {
                "ref": "96874",
                "color_code": "4",
                "color_hex": "#FE8A18",
                "color_name": "Orange",
                "qty": 1,
                "name": "Part 96874"
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 4,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 4,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 2,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 2,
                "name": "Part 98138"
            }
        ]
    },
    "75105-1": {
        "name": "Millennium Falcon",
        "minifigures": [
            {
                "ref": "fig-001714",
                "name": "Chewbacca, Dark Brown - Long Lines on Legs on Inside",
                "qty": 1
            },
            {
                "ref": "fig-001810",
                "name": "Finn, Black Undershirt",
                "qty": 1
            },
            {
                "ref": "fig-002057",
                "name": "Rey, Dark Tan Robe",
                "qty": 1
            },
            {
                "ref": "fig-002544",
                "name": "BB-8",
                "qty": 1
            },
            {
                "ref": "fig-002546",
                "name": "Han Solo, Old, Smile",
                "qty": 1
            },
            {
                "ref": "fig-002547",
                "name": "Kanjiklub Gang Member",
                "qty": 1
            },
            {
                "ref": "fig-002548",
                "name": "Tasu Leech",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "11203",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 11203"
            },
            {
                "ref": "11211",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 3,
                "name": "Part 11211"
            },
            {
                "ref": "11212",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 11212"
            },
            {
                "ref": "11213",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 11213"
            },
            {
                "ref": "11477",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 4,
                "name": "Part 11477"
            },
            {
                "ref": "11833",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 11833"
            },
            {
                "ref": "14181",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 14181"
            },
            {
                "ref": "14301",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 14301"
            },
            {
                "ref": "14418",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 14418"
            },
            {
                "ref": "14769",
                "color_code": "157",
                "color_hex": "#AC78BA",
                "color_name": "Medium Lavender",
                "qty": 2,
                "name": "Part 14769"
            },
            {
                "ref": "14769",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 14769"
            },
            {
                "ref": "15303",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 3,
                "name": "Part 15303"
            },
            {
                "ref": "15392",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 15392"
            },
            {
                "ref": "15392",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 15392"
            },
            {
                "ref": "15400",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 15400"
            },
            {
                "ref": "15573",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 15573"
            },
            {
                "ref": "15573",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 15573"
            },
            {
                "ref": "15712",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 15,
                "name": "Part 15712"
            },
            {
                "ref": "16577",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 16577"
            },
            {
                "ref": "19798",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 19798"
            },
            {
                "ref": "20105",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 20105"
            },
            {
                "ref": "21537",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 21537"
            },
            {
                "ref": "2412b",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 20,
                "name": "Part 2412b"
            },
            {
                "ref": "2412b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 9,
                "name": "Part 2412b"
            },
            {
                "ref": "2419",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2419"
            },
            {
                "ref": "2420",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 2420"
            },
            {
                "ref": "2420",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 2420"
            },
            {
                "ref": "2431",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2431"
            },
            {
                "ref": "2431",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 2431"
            },
            {
                "ref": "2440",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 2440"
            },
            {
                "ref": "2445",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2445"
            },
            {
                "ref": "2445",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2445"
            },
            {
                "ref": "2453b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 2453b"
            },
            {
                "ref": "2456",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 16,
                "name": "Part 2456"
            },
            {
                "ref": "2460",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 2460"
            },
            {
                "ref": "2540",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 2540"
            },
            {
                "ref": "2561",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 2561"
            },
            {
                "ref": "2562",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 2562"
            },
            {
                "ref": "2562",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 2562"
            },
            {
                "ref": "2723",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 2723"
            },
            {
                "ref": "2736",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2736"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 2780"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 36,
                "name": "Part 2780"
            },
            {
                "ref": "2877",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 2877"
            },
            {
                "ref": "298c02",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 298c02"
            },
            {
                "ref": "298c02",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 298c02"
            },
            {
                "ref": "30000",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 30000"
            },
            {
                "ref": "3001",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Part 3001"
            },
            {
                "ref": "3001",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3001"
            },
            {
                "ref": "3001",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3001"
            },
            {
                "ref": "3001",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 3001"
            },
            {
                "ref": "3002",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3002"
            },
            {
                "ref": "3003",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 3003"
            },
            {
                "ref": "3003",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 1,
                "name": "Part 3003"
            },
            {
                "ref": "3004",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 11,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 5,
                "name": "Part 3004"
            },
            {
                "ref": "3005",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3005"
            },
            {
                "ref": "3009",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 31,
                "name": "Part 3009"
            },
            {
                "ref": "3010",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 3010"
            },
            {
                "ref": "3010",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 8,
                "name": "Part 3010"
            },
            {
                "ref": "30136",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 6,
                "name": "Part 30136"
            },
            {
                "ref": "3020",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 3020"
            },
            {
                "ref": "3020",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3020"
            },
            {
                "ref": "3020",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 2,
                "name": "Part 3020"
            },
            {
                "ref": "3020",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 19,
                "name": "Part 3020"
            },
            {
                "ref": "3021",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 44,
                "name": "Part 3021"
            },
            {
                "ref": "3021",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 3021"
            },
            {
                "ref": "3021",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 3021"
            },
            {
                "ref": "3021",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 2,
                "name": "Part 3021"
            },
            {
                "ref": "3022",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 1,
                "name": "Part 3022"
            },
            {
                "ref": "3022",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 23,
                "name": "Part 3022"
            },
            {
                "ref": "3022",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 55,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 21,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 14,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 4,
                "name": "Part 3023"
            },
            {
                "ref": "30236",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 30236"
            },
            {
                "ref": "3024",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3024"
            },
            {
                "ref": "3027",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 3027"
            },
            {
                "ref": "3031",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 5,
                "name": "Part 3031"
            },
            {
                "ref": "3032",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3032"
            },
            {
                "ref": "3032",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3032"
            },
            {
                "ref": "3033",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3033"
            },
            {
                "ref": "3034",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3034"
            },
            {
                "ref": "3035",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3035"
            },
            {
                "ref": "30350b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 30350b"
            },
            {
                "ref": "30355",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 30355"
            },
            {
                "ref": "30356",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 30356"
            },
            {
                "ref": "3039",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 3039"
            },
            {
                "ref": "3040b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3040b"
            },
            {
                "ref": "3040b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3040b"
            },
            {
                "ref": "30414",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 30414"
            },
            {
                "ref": "3043",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3043"
            },
            {
                "ref": "30504",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 30504"
            },
            {
                "ref": "30565",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 30565"
            },
            {
                "ref": "3062b",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3062b"
            },
            {
                "ref": "3062b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3062b"
            },
            {
                "ref": "3062b",
                "color_code": "68",
                "color_hex": "#A95500",
                "color_name": "Dark Orange",
                "qty": 2,
                "name": "Part 3062b"
            },
            {
                "ref": "3068b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 3068b"
            },
            {
                "ref": "3069b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 3,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 8,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 11,
                "name": "Part 3069b"
            },
            {
                "ref": "3069bpr0070",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3069bpr0070"
            },
            {
                "ref": "32000",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 32000"
            },
            {
                "ref": "32028",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 32028"
            },
            {
                "ref": "32028",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 32028"
            },
            {
                "ref": "32054",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 32054"
            },
            {
                "ref": "32064a",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 32064a"
            },
            {
                "ref": "32140",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 32140"
            },
            {
                "ref": "3245c",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3245c"
            },
            {
                "ref": "32524",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 32524"
            },
            {
                "ref": "32526",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 32526"
            },
            {
                "ref": "32531",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 32531"
            },
            {
                "ref": "32532",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 32532"
            },
            {
                "ref": "3460",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3460"
            },
            {
                "ref": "3460",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 6,
                "name": "Part 3460"
            },
            {
                "ref": "3622",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3622"
            },
            {
                "ref": "3623",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 10,
                "name": "Part 3623"
            },
            {
                "ref": "3623",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 3623"
            },
            {
                "ref": "3623",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 4,
                "name": "Part 3623"
            },
            {
                "ref": "3665",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 24,
                "name": "Part 3665"
            },
            {
                "ref": "3666",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 5,
                "name": "Part 3666"
            },
            {
                "ref": "3666",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 8,
                "name": "Part 3666"
            },
            {
                "ref": "3678b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 3678b"
            },
            {
                "ref": "3678b",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 4,
                "name": "Part 3678b"
            },
            {
                "ref": "3678b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 7,
                "name": "Part 3678b"
            },
            {
                "ref": "3679",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 3679"
            },
            {
                "ref": "3680",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 3680"
            },
            {
                "ref": "3700",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 3700"
            },
            {
                "ref": "3701",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3701"
            },
            {
                "ref": "3702",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3702"
            },
            {
                "ref": "3703",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3703"
            },
            {
                "ref": "3709",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 3709"
            },
            {
                "ref": "3710",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 2,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 6,
                "name": "Part 3710"
            },
            {
                "ref": "3747b",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 6,
                "name": "Part 3747b"
            },
            {
                "ref": "3747b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3747b"
            },
            {
                "ref": "3795",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 13,
                "name": "Part 3795"
            },
            {
                "ref": "3795",
                "color_code": "69",
                "color_hex": "#958A73",
                "color_name": "Dark Tan",
                "qty": 4,
                "name": "Part 3795"
            },
            {
                "ref": "3795",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 3795"
            },
            {
                "ref": "3795",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3795"
            },
            {
                "ref": "3832",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 3832"
            },
            {
                "ref": "3832",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3832"
            },
            {
                "ref": "3839b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3839b"
            },
            {
                "ref": "3894",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 3894"
            },
            {
                "ref": "3895",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 10,
                "name": "Part 3895"
            },
            {
                "ref": "3937",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3937"
            },
            {
                "ref": "3938",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3938"
            },
            {
                "ref": "3958",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3958"
            },
            {
                "ref": "3958",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3958"
            },
            {
                "ref": "3960pr0008",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 3960pr0008"
            },
            {
                "ref": "4006",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 4006"
            },
            {
                "ref": "4032a",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4032a"
            },
            {
                "ref": "4032a",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 4032a"
            },
            {
                "ref": "4070",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 4070"
            },
            {
                "ref": "41539",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 41539"
            },
            {
                "ref": "4162",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 4162"
            },
            {
                "ref": "4175",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 4175"
            },
            {
                "ref": "41767",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 41767"
            },
            {
                "ref": "41768",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 41768"
            },
            {
                "ref": "4287c",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 4287c"
            },
            {
                "ref": "43093",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 43093"
            },
            {
                "ref": "4345b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 4345b"
            },
            {
                "ref": "4346",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 4346"
            },
            {
                "ref": "43722",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 43722"
            },
            {
                "ref": "43723",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 43723"
            },
            {
                "ref": "4445",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 4445"
            },
            {
                "ref": "4460b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 4460b"
            },
            {
                "ref": "44728",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 13,
                "name": "Part 44728"
            },
            {
                "ref": "44728",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 44728"
            },
            {
                "ref": "4510",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 4510"
            },
            {
                "ref": "4519",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 4519"
            },
            {
                "ref": "4590",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 4590"
            },
            {
                "ref": "4599b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 4599b"
            },
            {
                "ref": "4599b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4599b"
            },
            {
                "ref": "4735",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 4735"
            },
            {
                "ref": "47397",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 47397"
            },
            {
                "ref": "47397",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 47397"
            },
            {
                "ref": "47398",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 47398"
            },
            {
                "ref": "47398",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 47398"
            },
            {
                "ref": "4740",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4740"
            },
            {
                "ref": "47543",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 47543"
            },
            {
                "ref": "47543pr0006",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 47543pr0006"
            },
            {
                "ref": "48336",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 48336"
            },
            {
                "ref": "4865b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 4865b"
            },
            {
                "ref": "4865b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 4865b"
            },
            {
                "ref": "50950",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 14,
                "name": "Part 50950"
            },
            {
                "ref": "51739",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 51739"
            },
            {
                "ref": "51739",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 5,
                "name": "Part 51739"
            },
            {
                "ref": "54383",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 54383"
            },
            {
                "ref": "54383",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 54383"
            },
            {
                "ref": "54384",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 54384"
            },
            {
                "ref": "54384",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 54384"
            },
            {
                "ref": "58247",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 58247"
            },
            {
                "ref": "59349",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 59349"
            },
            {
                "ref": "59900",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 59900"
            },
            {
                "ref": "60208",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 60208"
            },
            {
                "ref": "60474",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 60474"
            },
            {
                "ref": "60477",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 60477"
            },
            {
                "ref": "60478",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 40,
                "name": "Part 60478"
            },
            {
                "ref": "60479",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 60479"
            },
            {
                "ref": "60897",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 60897"
            },
            {
                "ref": "6091",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 6091"
            },
            {
                "ref": "6106",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 6106"
            },
            {
                "ref": "6112",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 6112"
            },
            {
                "ref": "61184",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 61184"
            },
            {
                "ref": "61252",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 61252"
            },
            {
                "ref": "61252",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 61252"
            },
            {
                "ref": "6134",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 6134"
            },
            {
                "ref": "61409",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 61409"
            },
            {
                "ref": "6141",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 9,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 5,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 9,
                "name": "Part 6141"
            },
            {
                "ref": "61485",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 61485"
            },
            {
                "ref": "6177b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 6177b"
            },
            {
                "ref": "61780",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 61780"
            },
            {
                "ref": "6179",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 6179"
            },
            {
                "ref": "6179",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 6179"
            },
            {
                "ref": "6190",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 6190"
            },
            {
                "ref": "62462",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 62462"
            },
            {
                "ref": "6249",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 14,
                "name": "Part 6249"
            },
            {
                "ref": "63864",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 10,
                "name": "Part 63864"
            },
            {
                "ref": "63868",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 58,
                "name": "Part 63868"
            },
            {
                "ref": "63868",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 63868"
            },
            {
                "ref": "63965",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 63965"
            },
            {
                "ref": "63965",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 63965"
            },
            {
                "ref": "6541",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 6541"
            },
            {
                "ref": "6558",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 6,
                "name": "Part 6558"
            },
            {
                "ref": "6628",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 6628"
            },
            {
                "ref": "6636",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 6636"
            },
            {
                "ref": "6636",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 36,
                "name": "Part 6636"
            },
            {
                "ref": "75902pr0007",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 75902pr0007"
            },
            {
                "ref": "75937",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 2,
                "name": "Part 75937"
            },
            {
                "ref": "78c19",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 78c19"
            },
            {
                "ref": "85984",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 85984"
            },
            {
                "ref": "85984",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 7,
                "name": "Part 85984"
            },
            {
                "ref": "85984",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 85984"
            },
            {
                "ref": "87079",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 87079"
            },
            {
                "ref": "87079",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 87079"
            },
            {
                "ref": "87081",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 87081"
            },
            {
                "ref": "87087",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 87087"
            },
            {
                "ref": "87580",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 87580"
            },
            {
                "ref": "87609",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 87609"
            },
            {
                "ref": "87618",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 87618"
            },
            {
                "ref": "87620",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 8,
                "name": "Part 87620"
            },
            {
                "ref": "88072",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 88072"
            },
            {
                "ref": "88072",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 5,
                "name": "Part 88072"
            },
            {
                "ref": "88930",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 88930"
            },
            {
                "ref": "90258",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 90258"
            },
            {
                "ref": "92099",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 14,
                "name": "Part 92099"
            },
            {
                "ref": "92280",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 92280"
            },
            {
                "ref": "92280",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 92280"
            },
            {
                "ref": "92280",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 92280"
            },
            {
                "ref": "92438",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 92438"
            },
            {
                "ref": "92593",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 92593"
            },
            {
                "ref": "92738",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 92738"
            },
            {
                "ref": "92738",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 1,
                "name": "Part 92738"
            },
            {
                "ref": "92947",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 92947"
            },
            {
                "ref": "93273",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 93273"
            },
            {
                "ref": "93274",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 93274"
            },
            {
                "ref": "96874",
                "color_code": "4",
                "color_hex": "#FE8A18",
                "color_name": "Orange",
                "qty": 1,
                "name": "Part 96874"
            },
            {
                "ref": "98138",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 4,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 2,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "95",
                "color_hex": "#898788",
                "color_name": "Flat Silver",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "99207",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 99207"
            },
            {
                "ref": "99780",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 99780"
            }
        ]
    },
    "60093-1": {
        "name": "Deep Sea Helicopter",
        "minifigures": [
            {
                "ref": "fig-008027",
                "name": "Fireman, Red Jacket with Straps, Dark Blue Legs, White Helmet with Visor, Orange Sunglasses",
                "qty": 1
            },
            {
                "ref": "fig-009755",
                "name": "Diver Adam, Black and Red Wetsuit with Gauges, Red Helmet with Mask",
                "qty": 1
            },
            {
                "ref": "fig-009757",
                "name": "Marine Explorer, Woman, Open Red Jacket over Dark Red Sweater, Reddish Brown Hair, Glasses",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "10190",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 10190"
            },
            {
                "ref": "11211",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 11211"
            },
            {
                "ref": "11293",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 1,
                "name": "Part 11293"
            },
            {
                "ref": "11295",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 11295"
            },
            {
                "ref": "11297",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 11297"
            },
            {
                "ref": "11476",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 11476"
            },
            {
                "ref": "14518",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 14518"
            },
            {
                "ref": "15068",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 3,
                "name": "Part 15068"
            },
            {
                "ref": "15254",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 15254"
            },
            {
                "ref": "15712",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 15712"
            },
            {
                "ref": "17485",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 17485"
            },
            {
                "ref": "18990",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 18990"
            },
            {
                "ref": "19220",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 19220"
            },
            {
                "ref": "20512pr0001",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 20512pr0001"
            },
            {
                "ref": "20802",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 20802"
            },
            {
                "ref": "21465",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 21465"
            },
            {
                "ref": "2357",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 2357"
            },
            {
                "ref": "2412b",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 3,
                "name": "Part 2412b"
            },
            {
                "ref": "2412b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 2412b"
            },
            {
                "ref": "2431",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Part 2431"
            },
            {
                "ref": "2445",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 2445"
            },
            {
                "ref": "2447",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Part 2447"
            },
            {
                "ref": "2460",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 2460"
            },
            {
                "ref": "2476a",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 2476a"
            },
            {
                "ref": "2584",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2584"
            },
            {
                "ref": "2585",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2585"
            },
            {
                "ref": "2877",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 2877"
            },
            {
                "ref": "3001",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3001"
            },
            {
                "ref": "3001",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 3001"
            },
            {
                "ref": "3003",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3003"
            },
            {
                "ref": "3004",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3004"
            },
            {
                "ref": "3005",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 3005"
            },
            {
                "ref": "3005",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3005"
            },
            {
                "ref": "3009",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3009"
            },
            {
                "ref": "30090",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 30090"
            },
            {
                "ref": "3010",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 3010"
            },
            {
                "ref": "30150",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 1,
                "name": "Part 30150"
            },
            {
                "ref": "30153",
                "color_code": "19",
                "color_hex": "#F5CD2F",
                "color_name": "Trans-Yellow",
                "qty": 1,
                "name": "Part 30153"
            },
            {
                "ref": "30153",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 1,
                "name": "Part 30153"
            },
            {
                "ref": "3020",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 3020"
            },
            {
                "ref": "3020",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 3020"
            },
            {
                "ref": "3021",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3021"
            },
            {
                "ref": "3021",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Part 3021"
            },
            {
                "ref": "3022",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 3022"
            },
            {
                "ref": "3022",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 6,
                "name": "Part 3023"
            },
            {
                "ref": "3030",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3030"
            },
            {
                "ref": "3031",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3031"
            },
            {
                "ref": "3031",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3031"
            },
            {
                "ref": "3036",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3036"
            },
            {
                "ref": "30361c",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 30361c"
            },
            {
                "ref": "30367c",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 30367c"
            },
            {
                "ref": "3037",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 3037"
            },
            {
                "ref": "30383",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 30383"
            },
            {
                "ref": "30395",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 30395"
            },
            {
                "ref": "3039pr0013",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3039pr0013"
            },
            {
                "ref": "3040b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3040b"
            },
            {
                "ref": "30554b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 30554b"
            },
            {
                "ref": "30592",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 30592"
            },
            {
                "ref": "3068b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 3068b"
            },
            {
                "ref": "3069bpr0101",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3069bpr0101"
            },
            {
                "ref": "3070b",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 2,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "20",
                "color_hex": "#84B68D",
                "color_name": "Trans-Green",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "20",
                "color_hex": "#84B68D",
                "color_name": "Trans-Green",
                "qty": 2,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3176",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 3176"
            },
            {
                "ref": "32059",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 32059"
            },
            {
                "ref": "32125",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 32125"
            },
            {
                "ref": "32270",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 32270"
            },
            {
                "ref": "3298",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3298"
            },
            {
                "ref": "3460",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3460"
            },
            {
                "ref": "3460",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Part 3460"
            },
            {
                "ref": "3623",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 10,
                "name": "Part 3623"
            },
            {
                "ref": "3665",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3665"
            },
            {
                "ref": "3666",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3666"
            },
            {
                "ref": "3673",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3673"
            },
            {
                "ref": "3673",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3673"
            },
            {
                "ref": "3700",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3700"
            },
            {
                "ref": "3710",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3710"
            },
            {
                "ref": "3713",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 3713"
            },
            {
                "ref": "3713",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 3713"
            },
            {
                "ref": "3747b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 3747b"
            },
            {
                "ref": "3749",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3749"
            },
            {
                "ref": "3795",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3795"
            },
            {
                "ref": "3894",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3894"
            },
            {
                "ref": "3956",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3956"
            },
            {
                "ref": "4032a",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 4032a"
            },
            {
                "ref": "4032a",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 4032a"
            },
            {
                "ref": "4079b",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 4079b"
            },
            {
                "ref": "4081b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 4081b"
            },
            {
                "ref": "4081b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 4081b"
            },
            {
                "ref": "41529",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 41529"
            },
            {
                "ref": "41532",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 41532"
            },
            {
                "ref": "4175",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4175"
            },
            {
                "ref": "42023",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 42023"
            },
            {
                "ref": "43093",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 43093"
            },
            {
                "ref": "43720",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 43720"
            },
            {
                "ref": "43721",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 43721"
            },
            {
                "ref": "43857",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 43857"
            },
            {
                "ref": "4477",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 4477"
            },
            {
                "ref": "4624",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4624"
            },
            {
                "ref": "4624",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 4624"
            },
            {
                "ref": "4868b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 4868b"
            },
            {
                "ref": "4869",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 4869"
            },
            {
                "ref": "4871",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 4871"
            },
            {
                "ref": "48729b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 48729b"
            },
            {
                "ref": "48729b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 48729b"
            },
            {
                "ref": "50943",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 50943"
            },
            {
                "ref": "50950",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 50950"
            },
            {
                "ref": "51739",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 51739"
            },
            {
                "ref": "52501",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 52501"
            },
            {
                "ref": "54200",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 54200"
            },
            {
                "ref": "55013",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 55013"
            },
            {
                "ref": "56823c50",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 56823c50"
            },
            {
                "ref": "57906",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 6,
                "name": "Part 57906"
            },
            {
                "ref": "59895",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 8,
                "name": "Part 59895"
            },
            {
                "ref": "59895",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 59895"
            },
            {
                "ref": "60219",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 60219"
            },
            {
                "ref": "6041",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 6041"
            },
            {
                "ref": "60471",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 60471"
            },
            {
                "ref": "60478",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 60478"
            },
            {
                "ref": "60479",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 60479"
            },
            {
                "ref": "60479",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 60479"
            },
            {
                "ref": "60601",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 8,
                "name": "Part 60601"
            },
            {
                "ref": "6081",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 10,
                "name": "Part 6081"
            },
            {
                "ref": "6091",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 6091"
            },
            {
                "ref": "6091",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 6091"
            },
            {
                "ref": "6091",
                "color_code": "63",
                "color_hex": "#0A3463",
                "color_name": "Dark Blue",
                "qty": 4,
                "name": "Part 6091"
            },
            {
                "ref": "61345",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 61345"
            },
            {
                "ref": "61409",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 61409"
            },
            {
                "ref": "6141",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 6141"
            },
            {
                "ref": "61483",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 61483"
            },
            {
                "ref": "6183",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 3,
                "name": "Part 6183"
            },
            {
                "ref": "6192b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 6192b"
            },
            {
                "ref": "62462",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 62462"
            },
            {
                "ref": "63868",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 63868"
            },
            {
                "ref": "64567",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 64567"
            },
            {
                "ref": "64567",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 64567"
            },
            {
                "ref": "64799",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 64799"
            },
            {
                "ref": "6541",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 6541"
            },
            {
                "ref": "6583",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 6583"
            },
            {
                "ref": "6636",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 6636"
            },
            {
                "ref": "73590c03a",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 73590c03a"
            },
            {
                "ref": "74698",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 74698"
            },
            {
                "ref": "85984",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 85984"
            },
            {
                "ref": "87079",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 87079"
            },
            {
                "ref": "87552",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 87552"
            },
            {
                "ref": "92099",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 92099"
            },
            {
                "ref": "92593",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 92593"
            },
            {
                "ref": "95120",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 95120"
            },
            {
                "ref": "98138",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 3,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 4,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98835",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 98835"
            },
            {
                "ref": "99206",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 99206"
            },
            {
                "ref": "99207",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 99207"
            },
            {
                "ref": "99780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 99780"
            },
            {
                "ref": "99781",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 99781"
            }
        ]
    },
    "8088-1": {
        "name": "ARC-170 Starfighter",
        "minifigures": [
            {
                "ref": "fig-003690",
                "name": "Kit Fisto",
                "qty": 1
            },
            {
                "ref": "fig-003839",
                "name": "Astromech Droid, R4-P44, Pearl Gray print",
                "qty": 1
            },
            {
                "ref": "fig-003840",
                "name": "Clone Captain Jag",
                "qty": 1
            },
            {
                "ref": "fig-003841",
                "name": "Clone Pilot, Open Helmet with Black Markings",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "2412b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 5,
                "name": "Part 2412b"
            },
            {
                "ref": "2444",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 12,
                "name": "Part 2444"
            },
            {
                "ref": "2445",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 2445"
            },
            {
                "ref": "2456",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2456"
            },
            {
                "ref": "2515",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 2515"
            },
            {
                "ref": "2654",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 2654"
            },
            {
                "ref": "2730",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 2730"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 14,
                "name": "Part 2780"
            },
            {
                "ref": "2780",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 2780"
            },
            {
                "ref": "2877",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 2877"
            },
            {
                "ref": "2877",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 9,
                "name": "Part 2877"
            },
            {
                "ref": "3001",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3001"
            },
            {
                "ref": "3003",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3003"
            },
            {
                "ref": "3004",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 4,
                "name": "Part 3004"
            },
            {
                "ref": "3009",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 3009"
            },
            {
                "ref": "3020",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 5,
                "name": "Part 3020"
            },
            {
                "ref": "3021",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 3021"
            },
            {
                "ref": "3022",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 13,
                "name": "Part 3023"
            },
            {
                "ref": "3031",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 3031"
            },
            {
                "ref": "3032",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3032"
            },
            {
                "ref": "3034",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 3034"
            },
            {
                "ref": "3035",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3035"
            },
            {
                "ref": "30357",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 30357"
            },
            {
                "ref": "30361c",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 30361c"
            },
            {
                "ref": "30367b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 30367b"
            },
            {
                "ref": "3037",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 4,
                "name": "Part 3037"
            },
            {
                "ref": "30374",
                "color_code": "16",
                "color_hex": "#F8F184",
                "color_name": "Trans-Neon Green",
                "qty": 1,
                "name": "Part 30374"
            },
            {
                "ref": "3038",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3038"
            },
            {
                "ref": "3039",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3039"
            },
            {
                "ref": "30552",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 30552"
            },
            {
                "ref": "30553",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 30553"
            },
            {
                "ref": "3176",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 7,
                "name": "Part 3176"
            },
            {
                "ref": "32000",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 32000"
            },
            {
                "ref": "32013",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 32013"
            },
            {
                "ref": "32028",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 32028"
            },
            {
                "ref": "32062",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 32062"
            },
            {
                "ref": "32123b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 32123b"
            },
            {
                "ref": "32123b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Part 32123b"
            },
            {
                "ref": "32140",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 32140"
            },
            {
                "ref": "32184",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 32184"
            },
            {
                "ref": "3460",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 6,
                "name": "Part 3460"
            },
            {
                "ref": "3648b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3648b"
            },
            {
                "ref": "3659",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 3659"
            },
            {
                "ref": "3660",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 7,
                "name": "Part 3660"
            },
            {
                "ref": "3665",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 3665"
            },
            {
                "ref": "3666",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3666"
            },
            {
                "ref": "3700",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3700"
            },
            {
                "ref": "3707",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 3707"
            },
            {
                "ref": "3710",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 3710"
            },
            {
                "ref": "3713",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3713"
            },
            {
                "ref": "3713",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3713"
            },
            {
                "ref": "3737",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 7,
                "name": "Part 3737"
            },
            {
                "ref": "3794b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 3794b"
            },
            {
                "ref": "3795",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 6,
                "name": "Part 3795"
            },
            {
                "ref": "3830",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 4,
                "name": "Part 3830"
            },
            {
                "ref": "3831",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 4,
                "name": "Part 3831"
            },
            {
                "ref": "3937",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 3937"
            },
            {
                "ref": "3941",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 5,
                "name": "Part 3941"
            },
            {
                "ref": "4019",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 5,
                "name": "Part 4019"
            },
            {
                "ref": "4032a",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 4032a"
            },
            {
                "ref": "40490",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 40490"
            },
            {
                "ref": "4150",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 3,
                "name": "Part 4150"
            },
            {
                "ref": "41531",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 41531"
            },
            {
                "ref": "41531",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 41531"
            },
            {
                "ref": "41677",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 41677"
            },
            {
                "ref": "41764",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 41764"
            },
            {
                "ref": "41765",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 41765"
            },
            {
                "ref": "41769",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 41769"
            },
            {
                "ref": "41770",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 41770"
            },
            {
                "ref": "41896",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 41896"
            },
            {
                "ref": "42023",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 42023"
            },
            {
                "ref": "4274",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 4274"
            },
            {
                "ref": "4274",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 4274"
            },
            {
                "ref": "4287b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 4287b"
            },
            {
                "ref": "43093",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 43093"
            },
            {
                "ref": "43337",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 43337"
            },
            {
                "ref": "43710",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 43710"
            },
            {
                "ref": "43711",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 43711"
            },
            {
                "ref": "43720",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 43720"
            },
            {
                "ref": "43721",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 43721"
            },
            {
                "ref": "44728",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 44728"
            },
            {
                "ref": "4519",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 9,
                "name": "Part 4519"
            },
            {
                "ref": "4589",
                "color_code": "20",
                "color_hex": "#84B68D",
                "color_name": "Trans-Green",
                "qty": 2,
                "name": "Part 4589"
            },
            {
                "ref": "4716",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4716"
            },
            {
                "ref": "47397",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 5,
                "name": "Part 47397"
            },
            {
                "ref": "47398",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 5,
                "name": "Part 47398"
            },
            {
                "ref": "4740",
                "color_code": "50",
                "color_hex": "#DF6695",
                "color_name": "Trans-Dark Pink",
                "qty": 2,
                "name": "Part 4740"
            },
            {
                "ref": "50304",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 50304"
            },
            {
                "ref": "50305",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 50305"
            },
            {
                "ref": "50950",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 50950"
            },
            {
                "ref": "51739",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 51739"
            },
            {
                "ref": "54383",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 1,
                "name": "Part 54383"
            },
            {
                "ref": "54384",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 1,
                "name": "Part 54384"
            },
            {
                "ref": "55982",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 55982"
            },
            {
                "ref": "58181",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 58181"
            },
            {
                "ref": "59443",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 5,
                "name": "Part 59443"
            },
            {
                "ref": "60208",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 60208"
            },
            {
                "ref": "60483",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 60483"
            },
            {
                "ref": "6111",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 6111"
            },
            {
                "ref": "61184",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 61184"
            },
            {
                "ref": "6134",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 6134"
            },
            {
                "ref": "6141",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "61678",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 61678"
            },
            {
                "ref": "6180",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 6180"
            },
            {
                "ref": "63965",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 63965"
            },
            {
                "ref": "64567",
                "color_code": "67",
                "color_hex": "#A5A9B4",
                "color_name": "Metallic Silver",
                "qty": 1,
                "name": "Part 64567"
            },
            {
                "ref": "6541",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 6541"
            },
            {
                "ref": "6558",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 10,
                "name": "Part 6558"
            },
            {
                "ref": "6588",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 6588"
            },
            {
                "ref": "6636",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 6636"
            },
            {
                "ref": "85545",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 85545"
            },
            {
                "ref": "85984",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 6,
                "name": "Part 85984"
            },
            {
                "ref": "87079",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 11,
                "name": "Part 87079"
            },
            {
                "ref": "87082",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 87082"
            },
            {
                "ref": "88528",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 88528"
            },
            {
                "ref": "89762",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 3,
                "name": "Part 89762"
            }
        ]
    },
    "41104-1": {
        "name": "Pop Star Dressing Room",
        "minifigures": [
            {
                "ref": "fig-002394",
                "name": "Livi - Black Top, Pearl Gold Skirt",
                "qty": 1
            },
            {
                "ref": "fig-002395",
                "name": "Emma - Medium Lavender Vest, Dark Blue Skirt",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "11153",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 11153"
            },
            {
                "ref": "11477",
                "color_code": "71",
                "color_hex": "#923978",
                "color_name": "Magenta",
                "qty": 2,
                "name": "Part 11477"
            },
            {
                "ref": "11609",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 11609"
            },
            {
                "ref": "11609",
                "color_code": "110",
                "color_hex": "#F8BB3D",
                "color_name": "Bright Light Orange",
                "qty": 7,
                "name": "Part 11609"
            },
            {
                "ref": "11609",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 2,
                "name": "Part 11609"
            },
            {
                "ref": "11609",
                "color_code": "110",
                "color_hex": "#F8BB3D",
                "color_name": "Bright Light Orange",
                "qty": 1,
                "name": "Part 11609"
            },
            {
                "ref": "11610",
                "color_code": "47",
                "color_hex": "#C870A0",
                "color_name": "Dark Pink",
                "qty": 1,
                "name": "Part 11610"
            },
            {
                "ref": "11833",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 11833"
            },
            {
                "ref": "14769",
                "color_code": "157",
                "color_hex": "#AC78BA",
                "color_name": "Medium Lavender",
                "qty": 3,
                "name": "Part 14769"
            },
            {
                "ref": "14769",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 14769"
            },
            {
                "ref": "15573",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 15573"
            },
            {
                "ref": "15573",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 6,
                "name": "Part 15573"
            },
            {
                "ref": "15677",
                "color_code": "153",
                "color_hex": "#078BC9",
                "color_name": "Dark Azure",
                "qty": 1,
                "name": "Part 15677"
            },
            {
                "ref": "15706",
                "color_code": "157",
                "color_hex": "#AC78BA",
                "color_name": "Medium Lavender",
                "qty": 4,
                "name": "Part 15706"
            },
            {
                "ref": "15712",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 2,
                "name": "Part 15712"
            },
            {
                "ref": "20482",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 20482"
            },
            {
                "ref": "20482",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 3,
                "name": "Part 20482"
            },
            {
                "ref": "21008pr0001",
                "color_code": "71",
                "color_hex": "#923978",
                "color_name": "Magenta",
                "qty": 1,
                "name": "Part 21008pr0001"
            },
            {
                "ref": "21008pr0001",
                "color_code": "153",
                "color_hex": "#078BC9",
                "color_name": "Dark Azure",
                "qty": 1,
                "name": "Part 21008pr0001"
            },
            {
                "ref": "21210",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 21210"
            },
            {
                "ref": "2357",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 2357"
            },
            {
                "ref": "2817",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2817"
            },
            {
                "ref": "3004",
                "color_code": "71",
                "color_hex": "#923978",
                "color_name": "Magenta",
                "qty": 11,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3004"
            },
            {
                "ref": "3005",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3005"
            },
            {
                "ref": "3005",
                "color_code": "71",
                "color_hex": "#923978",
                "color_name": "Magenta",
                "qty": 14,
                "name": "Part 3005"
            },
            {
                "ref": "3010",
                "color_code": "71",
                "color_hex": "#923978",
                "color_name": "Magenta",
                "qty": 7,
                "name": "Part 3010"
            },
            {
                "ref": "3010",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 11,
                "name": "Part 3010"
            },
            {
                "ref": "3020",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 3020"
            },
            {
                "ref": "3022",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 9,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "157",
                "color_hex": "#AC78BA",
                "color_name": "Medium Lavender",
                "qty": 5,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 5,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "19",
                "color_hex": "#F5CD2F",
                "color_name": "Trans-Yellow",
                "qty": 1,
                "name": "Part 3023"
            },
            {
                "ref": "3024",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 5,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3024"
            },
            {
                "ref": "3032",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3032"
            },
            {
                "ref": "3034",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3034"
            },
            {
                "ref": "30395",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 30395"
            },
            {
                "ref": "3068b",
                "color_code": "110",
                "color_hex": "#F8BB3D",
                "color_name": "Bright Light Orange",
                "qty": 2,
                "name": "Part 3068b"
            },
            {
                "ref": "3068bpr0255",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3068bpr0255"
            },
            {
                "ref": "3069b",
                "color_code": "156",
                "color_hex": "#36AEBF",
                "color_name": "Medium Azure",
                "qty": 3,
                "name": "Part 3069b"
            },
            {
                "ref": "3069bpr0055",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3069bpr0055"
            },
            {
                "ref": "32054",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 32054"
            },
            {
                "ref": "3245c",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3245c"
            },
            {
                "ref": "33051",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 33051"
            },
            {
                "ref": "33291",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 3,
                "name": "Part 33291"
            },
            {
                "ref": "33291",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 33291"
            },
            {
                "ref": "3623",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 7,
                "name": "Part 3623"
            },
            {
                "ref": "3623",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3623"
            },
            {
                "ref": "3626c",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3626c"
            },
            {
                "ref": "3660",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3660"
            },
            {
                "ref": "3666",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3666"
            },
            {
                "ref": "3666",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 10,
                "name": "Part 3666"
            },
            {
                "ref": "3709",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3709"
            },
            {
                "ref": "3710",
                "color_code": "157",
                "color_hex": "#AC78BA",
                "color_name": "Medium Lavender",
                "qty": 1,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3710"
            },
            {
                "ref": "3741",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 1,
                "name": "Part 3741"
            },
            {
                "ref": "3741",
                "color_code": "6",
                "color_hex": "#237841",
                "color_name": "Green",
                "qty": 1,
                "name": "Part 3741"
            },
            {
                "ref": "3795",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3795"
            },
            {
                "ref": "3795",
                "color_code": "157",
                "color_hex": "#AC78BA",
                "color_name": "Medium Lavender",
                "qty": 1,
                "name": "Part 3795"
            },
            {
                "ref": "3830",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3830"
            },
            {
                "ref": "3831",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3831"
            },
            {
                "ref": "3832",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3832"
            },
            {
                "ref": "3941",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3941"
            },
            {
                "ref": "3958",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 8,
                "name": "Part 3958"
            },
            {
                "ref": "42446",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 42446"
            },
            {
                "ref": "42446",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 42446"
            },
            {
                "ref": "4345b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 4345b"
            },
            {
                "ref": "4346",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 4346"
            },
            {
                "ref": "4346",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 4346"
            },
            {
                "ref": "4536",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 4536"
            },
            {
                "ref": "48336",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 48336"
            },
            {
                "ref": "4865a",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 4865a"
            },
            {
                "ref": "50950",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 50950"
            },
            {
                "ref": "54200",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 54200"
            },
            {
                "ref": "59900",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 2,
                "name": "Part 59900"
            },
            {
                "ref": "59900",
                "color_code": "14",
                "color_hex": "#0020A0",
                "color_name": "Trans-Dark Blue",
                "qty": 1,
                "name": "Part 59900"
            },
            {
                "ref": "60470b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 60470b"
            },
            {
                "ref": "60474",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 60474"
            },
            {
                "ref": "60581",
                "color_code": "71",
                "color_hex": "#923978",
                "color_name": "Magenta",
                "qty": 6,
                "name": "Part 60581"
            },
            {
                "ref": "60596",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 60596"
            },
            {
                "ref": "60616b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 60616b"
            },
            {
                "ref": "6111",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 3,
                "name": "Part 6111"
            },
            {
                "ref": "6141",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "19",
                "color_hex": "#F5CD2F",
                "color_name": "Trans-Yellow",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "19",
                "color_hex": "#F5CD2F",
                "color_name": "Trans-Yellow",
                "qty": 11,
                "name": "Part 6141"
            },
            {
                "ref": "61485",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 61485"
            },
            {
                "ref": "6180",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 6180"
            },
            {
                "ref": "6231",
                "color_code": "156",
                "color_hex": "#36AEBF",
                "color_name": "Medium Azure",
                "qty": 4,
                "name": "Part 6231"
            },
            {
                "ref": "6231",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 6231"
            },
            {
                "ref": "6256",
                "color_code": "110",
                "color_hex": "#F8BB3D",
                "color_name": "Bright Light Orange",
                "qty": 1,
                "name": "Part 6256"
            },
            {
                "ref": "63965",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 2,
                "name": "Part 63965"
            },
            {
                "ref": "73983",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 73983"
            },
            {
                "ref": "85080",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 6,
                "name": "Part 85080"
            },
            {
                "ref": "85984",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 1,
                "name": "Part 85984"
            },
            {
                "ref": "85984",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 85984"
            },
            {
                "ref": "87079",
                "color_code": "157",
                "color_hex": "#AC78BA",
                "color_name": "Medium Lavender",
                "qty": 2,
                "name": "Part 87079"
            },
            {
                "ref": "87079",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 87079"
            },
            {
                "ref": "87087",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 87087"
            },
            {
                "ref": "87544",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 2,
                "name": "Part 87544"
            },
            {
                "ref": "88072",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 88072"
            },
            {
                "ref": "92410",
                "color_code": "157",
                "color_hex": "#AC78BA",
                "color_name": "Medium Lavender",
                "qty": 1,
                "name": "Part 92410"
            },
            {
                "ref": "93088pr0003",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 93088pr0003"
            },
            {
                "ref": "93091",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 1,
                "name": "Part 93091"
            },
            {
                "ref": "93094pat0001",
                "color_code": "47",
                "color_hex": "#C870A0",
                "color_name": "Dark Pink",
                "qty": 1,
                "name": "Part 93094pat0001"
            },
            {
                "ref": "93094pat0001",
                "color_code": "47",
                "color_hex": "#C870A0",
                "color_name": "Dark Pink",
                "qty": 1,
                "name": "Part 93094pat0001"
            },
            {
                "ref": "93160",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 93160"
            },
            {
                "ref": "93160",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 93160"
            },
            {
                "ref": "96479",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 3,
                "name": "Part 96479"
            },
            {
                "ref": "96480",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 1,
                "name": "Part 96480"
            },
            {
                "ref": "96481",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 2,
                "name": "Part 96481"
            },
            {
                "ref": "96482",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 1,
                "name": "Part 96482"
            },
            {
                "ref": "96483",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 2,
                "name": "Part 96483"
            },
            {
                "ref": "96484",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 1,
                "name": "Part 96484"
            },
            {
                "ref": "96485",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 2,
                "name": "Part 96485"
            },
            {
                "ref": "96486",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 1,
                "name": "Part 96486"
            },
            {
                "ref": "96487",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 2,
                "name": "Part 96487"
            },
            {
                "ref": "96488",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 1,
                "name": "Part 96488"
            },
            {
                "ref": "96489",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 2,
                "name": "Part 96489"
            },
            {
                "ref": "96490",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 1,
                "name": "Part 96490"
            },
            {
                "ref": "96491",
                "color_code": "89",
                "color_hex": "#3F3691",
                "color_name": "Dark Purple",
                "qty": 1,
                "name": "Part 96491"
            },
            {
                "ref": "98138",
                "color_code": "51",
                "color_hex": "#A5A5CB",
                "color_name": "Trans-Purple",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "51",
                "color_hex": "#A5A5CB",
                "color_name": "Trans-Purple",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98549",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 98549"
            }
        ]
    },
    "60082-1": {
        "name": "Dune Buggy Trailer",
        "minifigures": [
            {
                "ref": "fig-007766",
                "name": "Woman, Open Green Jacket with Necklace, Dark Orange Hair",
                "qty": 1
            },
            {
                "ref": "fig-007767",
                "name": "Man, Open Blue Jacket over Dark Red Shirt, Sand Blue Legs, Red Cap, Stubble",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "10201",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 10201"
            },
            {
                "ref": "11477",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 11477"
            },
            {
                "ref": "11477",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 11477"
            },
            {
                "ref": "18892",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 18892"
            },
            {
                "ref": "19277",
                "color_code": "9999",
                "color_hex": "#05131D",
                "color_name": "[No Color/Any Color]",
                "qty": 1,
                "name": "Part 19277"
            },
            {
                "ref": "2412b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 2412b"
            },
            {
                "ref": "2432",
                "color_code": "88",
                "color_hex": "#582A12",
                "color_name": "Reddish Brown",
                "qty": 2,
                "name": "Part 2432"
            },
            {
                "ref": "2432",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2432"
            },
            {
                "ref": "2437",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Part 2437"
            },
            {
                "ref": "2445",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2445"
            },
            {
                "ref": "2446",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 2446"
            },
            {
                "ref": "2447",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Part 2447"
            },
            {
                "ref": "2447",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Part 2447"
            },
            {
                "ref": "2508",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2508"
            },
            {
                "ref": "2569",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2569"
            },
            {
                "ref": "2569",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2569"
            },
            {
                "ref": "2926",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 2926"
            },
            {
                "ref": "3004",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3004"
            },
            {
                "ref": "3009",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 3009"
            },
            {
                "ref": "3010",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3010"
            },
            {
                "ref": "30157",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 30157"
            },
            {
                "ref": "3020",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3020"
            },
            {
                "ref": "3020",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3020"
            },
            {
                "ref": "3021",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 5,
                "name": "Part 3021"
            },
            {
                "ref": "3022",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 5,
                "name": "Part 3022"
            },
            {
                "ref": "3022",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 3,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 3,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 4,
                "name": "Part 3023"
            },
            {
                "ref": "3024",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3024"
            },
            {
                "ref": "3024",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3024"
            },
            {
                "ref": "3029",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3029"
            },
            {
                "ref": "3031",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3031"
            },
            {
                "ref": "3032",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3032"
            },
            {
                "ref": "3032",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3032"
            },
            {
                "ref": "3034",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3034"
            },
            {
                "ref": "3035",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3035"
            },
            {
                "ref": "30350b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 30350b"
            },
            {
                "ref": "3036",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3036"
            },
            {
                "ref": "3062b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3062b"
            },
            {
                "ref": "3068b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3068b"
            },
            {
                "ref": "3069b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 3069b"
            },
            {
                "ref": "3069b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 3069b"
            },
            {
                "ref": "3070b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3070b"
            },
            {
                "ref": "3623",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 3623"
            },
            {
                "ref": "3623",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3623"
            },
            {
                "ref": "3660",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3660"
            },
            {
                "ref": "3665",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3665"
            },
            {
                "ref": "3666",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3666"
            },
            {
                "ref": "3666",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3666"
            },
            {
                "ref": "3666",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 3666"
            },
            {
                "ref": "3666",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 3666"
            },
            {
                "ref": "3710",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3710"
            },
            {
                "ref": "3829c01",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 3829c01"
            },
            {
                "ref": "3832",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 3832"
            },
            {
                "ref": "4006",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 4006"
            },
            {
                "ref": "4070",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 4070"
            },
            {
                "ref": "4081b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 4081b"
            },
            {
                "ref": "4081b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 4081b"
            },
            {
                "ref": "4083",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 4083"
            },
            {
                "ref": "4176",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 4176"
            },
            {
                "ref": "41769",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 41769"
            },
            {
                "ref": "41770",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 41770"
            },
            {
                "ref": "41854",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 41854"
            },
            {
                "ref": "44728",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 44728"
            },
            {
                "ref": "4488",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 4488"
            },
            {
                "ref": "45677",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 45677"
            },
            {
                "ref": "47457",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 47457"
            },
            {
                "ref": "47457",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 47457"
            },
            {
                "ref": "48336",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 48336"
            },
            {
                "ref": "50943",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 50943"
            },
            {
                "ref": "50950",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 50950"
            },
            {
                "ref": "50951",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 50951"
            },
            {
                "ref": "50951",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 50951"
            },
            {
                "ref": "54200",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 2,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "55981",
                "color_code": "115",
                "color_hex": "#AA7F2E",
                "color_name": "Pearl Gold",
                "qty": 4,
                "name": "Part 55981"
            },
            {
                "ref": "56890",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 56890"
            },
            {
                "ref": "59900",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 59900"
            },
            {
                "ref": "6014b",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 4,
                "name": "Part 6014b"
            },
            {
                "ref": "60219",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 60219"
            },
            {
                "ref": "6140",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 6140"
            },
            {
                "ref": "6141",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 14,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6180",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 6180"
            },
            {
                "ref": "63082",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 63082"
            },
            {
                "ref": "6636",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 6636"
            },
            {
                "ref": "85984",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 85984"
            },
            {
                "ref": "85984",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 85984"
            },
            {
                "ref": "87079",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 87079"
            },
            {
                "ref": "92402",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 4,
                "name": "Part 92402"
            },
            {
                "ref": "93274",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 93274"
            },
            {
                "ref": "93593",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 93593"
            },
            {
                "ref": "93606",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 93606"
            },
            {
                "ref": "98138",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 2,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 2,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 2,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "98",
                "color_hex": "#F08F1C",
                "color_name": "Trans-Orange",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98138",
                "color_code": "17",
                "color_hex": "#C91A09",
                "color_name": "Trans-Red",
                "qty": 1,
                "name": "Part 98138"
            },
            {
                "ref": "98282",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 98282"
            },
            {
                "ref": "98282",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 2,
                "name": "Part 98282"
            },
            {
                "ref": "99207",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 99207"
            },
            {
                "ref": "99781",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 99781"
            }
        ]
    },
    "6212-1": {
        "name": "X-wing Fighter",
        "minifigures": [
            {
                "ref": "fig-000124",
                "name": "Chewbacca, Reddish Brown",
                "qty": 1
            },
            {
                "ref": "fig-003536",
                "name": "Astromech Droid, R2-D2, White Dome, 30361c Body",
                "qty": 1
            },
            {
                "ref": "fig-003537",
                "name": "Princess Leia, Hoth Outfit, Tan Jacket, Reddish Brown Hair",
                "qty": 1
            },
            {
                "ref": "fig-003538",
                "name": "Luke Skywalker, Orange Rebel Pilot Outfit, Dark Bluish Gray Hips, Black Eyes",
                "qty": 1
            },
            {
                "ref": "fig-003539",
                "name": "Han Solo, Hoth Outfit, Dark Blue Jacket, Reddish Brown Legs",
                "qty": 1
            },
            {
                "ref": "fig-003540",
                "name": "Wedge Antilles, Plain Legs",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "23306",
                "color_code": "22",
                "color_hex": "#E0E0E0",
                "color_name": "Chrome Silver",
                "qty": 1,
                "name": "Part 23306"
            },
            {
                "ref": "2412b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 2412b"
            },
            {
                "ref": "2413",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 2413"
            },
            {
                "ref": "2420",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 8,
                "name": "Part 2420"
            },
            {
                "ref": "2431",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 2431"
            },
            {
                "ref": "2431",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 2431"
            },
            {
                "ref": "2431pr0027",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 2431pr0027"
            },
            {
                "ref": "2432",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 2432"
            },
            {
                "ref": "2445",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 2445"
            },
            {
                "ref": "2456",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 2456"
            },
            {
                "ref": "2654",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 2654"
            },
            {
                "ref": "2736",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 2736"
            },
            {
                "ref": "2877",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 8,
                "name": "Part 2877"
            },
            {
                "ref": "298c02",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 298c02"
            },
            {
                "ref": "298c02",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 298c02"
            },
            {
                "ref": "3001",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3001"
            },
            {
                "ref": "3003",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 3003"
            },
            {
                "ref": "3004",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 9,
                "name": "Part 3004"
            },
            {
                "ref": "3004",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 7,
                "name": "Part 3004"
            },
            {
                "ref": "3005",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 8,
                "name": "Part 3005"
            },
            {
                "ref": "3010",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 5,
                "name": "Part 3010"
            },
            {
                "ref": "30136",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 30136"
            },
            {
                "ref": "3020",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 7,
                "name": "Part 3020"
            },
            {
                "ref": "3020",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 3020"
            },
            {
                "ref": "3020",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3020"
            },
            {
                "ref": "3021",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3021"
            },
            {
                "ref": "3021",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3021"
            },
            {
                "ref": "3022",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 3022"
            },
            {
                "ref": "3022",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 6,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 5,
                "name": "Part 3023"
            },
            {
                "ref": "3023",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 6,
                "name": "Part 3023"
            },
            {
                "ref": "3031",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3031"
            },
            {
                "ref": "3032",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 3032"
            },
            {
                "ref": "3032",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3032"
            },
            {
                "ref": "3034",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3034"
            },
            {
                "ref": "30355",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 30355"
            },
            {
                "ref": "30356",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 30356"
            },
            {
                "ref": "30359b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 30359b"
            },
            {
                "ref": "30360",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 30360"
            },
            {
                "ref": "30364",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 30364"
            },
            {
                "ref": "30367b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 30367b"
            },
            {
                "ref": "3037",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3037"
            },
            {
                "ref": "30372",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 30372"
            },
            {
                "ref": "30372pr0001",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Part 30372pr0001"
            },
            {
                "ref": "30374",
                "color_code": "15",
                "color_hex": "#AEEFEC",
                "color_name": "Trans-Light Blue",
                "qty": 1,
                "name": "Part 30374"
            },
            {
                "ref": "30389b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 30389b"
            },
            {
                "ref": "3039",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 3039"
            },
            {
                "ref": "3039pr0008",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3039pr0008"
            },
            {
                "ref": "3040b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 3040b"
            },
            {
                "ref": "30414",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 30414"
            },
            {
                "ref": "30526",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 30526"
            },
            {
                "ref": "30553",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 30553"
            },
            {
                "ref": "3068b",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 3068b"
            },
            {
                "ref": "3068b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3068b"
            },
            {
                "ref": "3068bpr0071",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3068bpr0071"
            },
            {
                "ref": "3070b",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 6,
                "name": "Part 3070b"
            },
            {
                "ref": "3070b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 3070b"
            },
            {
                "ref": "3176",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 3176"
            },
            {
                "ref": "32000",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 18,
                "name": "Part 32000"
            },
            {
                "ref": "32123b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 32123b"
            },
            {
                "ref": "32123b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 32123b"
            },
            {
                "ref": "32124",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 1,
                "name": "Part 32124"
            },
            {
                "ref": "32269",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 32269"
            },
            {
                "ref": "32556a",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 32556a"
            },
            {
                "ref": "3460",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 3460"
            },
            {
                "ref": "3460",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 2,
                "name": "Part 3460"
            },
            {
                "ref": "3623",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 12,
                "name": "Part 3623"
            },
            {
                "ref": "3648b",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3648b"
            },
            {
                "ref": "3660",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 8,
                "name": "Part 3660"
            },
            {
                "ref": "3665",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3665"
            },
            {
                "ref": "3666",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3666"
            },
            {
                "ref": "3666",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 13,
                "name": "Part 3666"
            },
            {
                "ref": "3666",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3666"
            },
            {
                "ref": "3666",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 6,
                "name": "Part 3666"
            },
            {
                "ref": "3700",
                "color_code": "3",
                "color_hex": "#F2CD37",
                "color_name": "Yellow",
                "qty": 1,
                "name": "Part 3700"
            },
            {
                "ref": "3701",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 3701"
            },
            {
                "ref": "3706",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3706"
            },
            {
                "ref": "3707",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 3707"
            },
            {
                "ref": "3710",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 1,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3710"
            },
            {
                "ref": "3710",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 4,
                "name": "Part 3710"
            },
            {
                "ref": "3713",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3713"
            },
            {
                "ref": "3713",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3713"
            },
            {
                "ref": "3747a",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3747a"
            },
            {
                "ref": "3794a",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 3,
                "name": "Part 3794a"
            },
            {
                "ref": "3794a",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 3794a"
            },
            {
                "ref": "3795",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3795"
            },
            {
                "ref": "3830",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3830"
            },
            {
                "ref": "3830",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3830"
            },
            {
                "ref": "3831",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3831"
            },
            {
                "ref": "3831",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 3831"
            },
            {
                "ref": "3941",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 3941"
            },
            {
                "ref": "3941",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 3941"
            },
            {
                "ref": "4032a",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 6,
                "name": "Part 4032a"
            },
            {
                "ref": "4162",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 4162"
            },
            {
                "ref": "4162",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 4162"
            },
            {
                "ref": "41747",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 41747"
            },
            {
                "ref": "41748",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 41748"
            },
            {
                "ref": "41752",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 41752"
            },
            {
                "ref": "41769",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 41769"
            },
            {
                "ref": "41770",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 3,
                "name": "Part 41770"
            },
            {
                "ref": "42446",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 42446"
            },
            {
                "ref": "4287b",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 4287b"
            },
            {
                "ref": "43093",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 4,
                "name": "Part 43093"
            },
            {
                "ref": "43712",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 43712"
            },
            {
                "ref": "43713",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 43713"
            },
            {
                "ref": "43719",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 1,
                "name": "Part 43719"
            },
            {
                "ref": "44300",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 44300"
            },
            {
                "ref": "44568",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 44568"
            },
            {
                "ref": "44571",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 44571"
            },
            {
                "ref": "4477",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 4477"
            },
            {
                "ref": "4519",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 11,
                "name": "Part 4519"
            },
            {
                "ref": "4589",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 4589"
            },
            {
                "ref": "4716",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 4716"
            },
            {
                "ref": "4740",
                "color_code": "18",
                "color_hex": "#FF800D",
                "color_name": "Trans-Neon Orange",
                "qty": 4,
                "name": "Part 4740"
            },
            {
                "ref": "4740",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 4740"
            },
            {
                "ref": "4865a",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 4865a"
            },
            {
                "ref": "4871",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 4871"
            },
            {
                "ref": "6141",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 4,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "18",
                "color_hex": "#FF800D",
                "color_name": "Trans-Neon Orange",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "18",
                "color_hex": "#FF800D",
                "color_name": "Trans-Neon Orange",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 3,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6538b",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 10,
                "name": "Part 6538b"
            },
            {
                "ref": "6541",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 6541"
            },
            {
                "ref": "6553",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 6553"
            },
            {
                "ref": "6588",
                "color_code": "12",
                "color_hex": "#FCFCFC",
                "color_name": "Trans-Clear",
                "qty": 1,
                "name": "Part 6588"
            },
            {
                "ref": "6628",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 8,
                "name": "Part 6628"
            },
            {
                "ref": "6636",
                "color_code": "2",
                "color_hex": "#E4CD9E",
                "color_name": "Tan",
                "qty": 4,
                "name": "Part 6636"
            },
            {
                "ref": "6636",
                "color_code": "5",
                "color_hex": "#C91A09",
                "color_name": "Red",
                "qty": 2,
                "name": "Part 6636"
            },
            {
                "ref": "75535",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 75535"
            },
            {
                "ref": "76385",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 76385"
            },
            {
                "ref": "85543",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 6,
                "name": "Part 85543"
            }
        ]
    },
    "75072-1": {
        "name": "ARC-170 Starfighter",
        "minifigures": [
            {
                "ref": "fig-001287",
                "name": "Clone Pilot, Open Helmet with Yellow and Red Markings, Light Bluish Gray Legs (Episode 3)",
                "qty": 1
            }
        ],
        "parts": [
            {
                "ref": "11215",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 11215"
            },
            {
                "ref": "11458",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 11458"
            },
            {
                "ref": "11476",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 11476"
            },
            {
                "ref": "14769pr1005",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 14769pr1005"
            },
            {
                "ref": "15573",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 15573"
            },
            {
                "ref": "18677",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 18677"
            },
            {
                "ref": "2412b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 2412b"
            },
            {
                "ref": "2420",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 3,
                "name": "Part 2420"
            },
            {
                "ref": "2540",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2540"
            },
            {
                "ref": "2654",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 2,
                "name": "Part 2654"
            },
            {
                "ref": "298c02",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 298c02"
            },
            {
                "ref": "298c02",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 298c02"
            },
            {
                "ref": "3020",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 1,
                "name": "Part 3020"
            },
            {
                "ref": "3021",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3021"
            },
            {
                "ref": "3022",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3022"
            },
            {
                "ref": "3022",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 3,
                "name": "Part 3022"
            },
            {
                "ref": "3023",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 3023"
            },
            {
                "ref": "3031",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 3031"
            },
            {
                "ref": "3031",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 3031"
            },
            {
                "ref": "30602",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 1,
                "name": "Part 30602"
            },
            {
                "ref": "3176",
                "color_code": "320",
                "color_hex": "#720E0F",
                "color_name": "Dark Red",
                "qty": 2,
                "name": "Part 3176"
            },
            {
                "ref": "3623",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3623"
            },
            {
                "ref": "3665",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 3665"
            },
            {
                "ref": "4032a",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 4032a"
            },
            {
                "ref": "4032a",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 2,
                "name": "Part 4032a"
            },
            {
                "ref": "43093",
                "color_code": "7",
                "color_hex": "#0055BF",
                "color_name": "Blue",
                "qty": 2,
                "name": "Part 43093"
            },
            {
                "ref": "43722",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 43722"
            },
            {
                "ref": "43723",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 43723"
            },
            {
                "ref": "44676",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 4,
                "name": "Part 44676"
            },
            {
                "ref": "44728",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 44728"
            },
            {
                "ref": "51739",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 51739"
            },
            {
                "ref": "54200",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 6,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 3,
                "name": "Part 54200"
            },
            {
                "ref": "54200",
                "color_code": "13",
                "color_hex": "#635F52",
                "color_name": "Trans-Brown",
                "qty": 1,
                "name": "Part 54200"
            },
            {
                "ref": "58176",
                "color_code": "108",
                "color_hex": "#D9E4A7",
                "color_name": "Trans-Bright Green",
                "qty": 2,
                "name": "Part 58176"
            },
            {
                "ref": "59900",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 59900"
            },
            {
                "ref": "60470b",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 60470b"
            },
            {
                "ref": "60897",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 60897"
            },
            {
                "ref": "61184",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 4,
                "name": "Part 61184"
            },
            {
                "ref": "6141",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 4,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "108",
                "color_hex": "#D9E4A7",
                "color_name": "Trans-Bright Green",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "50",
                "color_hex": "#DF6695",
                "color_name": "Trans-Dark Pink",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "108",
                "color_hex": "#D9E4A7",
                "color_name": "Trans-Bright Green",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 6141"
            },
            {
                "ref": "6141",
                "color_code": "50",
                "color_hex": "#DF6695",
                "color_name": "Trans-Dark Pink",
                "qty": 2,
                "name": "Part 6141"
            },
            {
                "ref": "63965",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 2,
                "name": "Part 63965"
            },
            {
                "ref": "64799",
                "color_code": "86",
                "color_hex": "#A0A5A9",
                "color_name": "Light Bluish Gray",
                "qty": 1,
                "name": "Part 64799"
            },
            {
                "ref": "92738",
                "color_code": "11",
                "color_hex": "#05131D",
                "color_name": "Black",
                "qty": 1,
                "name": "Part 92738"
            },
            {
                "ref": "93273",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 93273"
            },
            {
                "ref": "99780",
                "color_code": "85",
                "color_hex": "#6C6E68",
                "color_name": "Dark Bluish Gray",
                "qty": 1,
                "name": "Part 99780"
            },
            {
                "ref": "99781",
                "color_code": "1",
                "color_hex": "#FFFFFF",
                "color_name": "White",
                "qty": 2,
                "name": "Part 99781"
            }
        ]
    }
}
