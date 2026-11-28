#pragma once
enum SongIndex { Sweet, Around, Pesenka, Smalltown, Lights, Blue, Love, Sunglasses, Hero, Eiffel,
                 Ann, Bell, Amor, Mirage, Mashup, Cry, HelloSong, Dream, Uzi, Parking, Headache, Cowgirl, Titanic, Doin, Hang, Superman, SongCount };
constexpr bool chantSuppliesArp(SongIndex song) { return song==Lights || song==Titanic; }
