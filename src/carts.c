// Cartridges whose save chip is a 16 Kbit EEPROM, by the two CRCs at 0x10 in the header (every other
// cartridge answers as a 4 Kbit EEPROM). The list is from mupen64plus's ROM database (GPLv2).
#include "n64.h"

static const uint64_t eeprom16k[] = {
	0x975B7845A2505C18ull, // 77a Special Edition by Count0 (PD)
	0x514B6900B4B19881ull, // Banjo to Kazooie no Daibouken 2 (J) [!]
	0x155B7CDFF0DA7325ull, // Banjo-Tooie (A) [!]
	0xC9176D39EA4779D1ull, // Banjo-Tooie (E) (M4) [!]
	0xC2E9AA9A475D70AAull, // Banjo-Tooie (U) [!]
	0x373F58899A6CA80Aull, // Conker's Bad Fur Day (E) [!]
	0x30C7AC507704072Dull, // Conker's Bad Fur Day (U) [!]
	0x83F3931ECB72223Dull, // Cruis'n World (E) [!]
	0xDFE61153D76118E6ull, // Cruis'n World (U) [!]
	0x11936D8C6F2C4B43ull, // Donkey Kong 64 (E) [!]
	0x053C89A7A5064302ull, // Donkey Kong 64 (J) [!]
	0x0DD4ABABB5A2A91Eull, // Donkey Kong 64 (U) (Kiosk Demo) [!]
	0xEC58EABFAD7C7169ull, // Donkey Kong 64 (U) [!]
	0xB6306E99B63ED2B2ull, // Doraemon 2 - Nobita to Hikari no Shinden (J) [!]
	0xA8275140B9B056E8ull, // Doraemon 3 - Nobita no Machi SOS! (J) [!]
	0x202A8EE483F88B89ull, // Excitebike 64 (E) [!]
	0x861C3519F6091CE5ull, // Excitebike 64 (J) [!]
	0xAF754F7B1DD17381ull, // Excitebike 64 (U) (Kiosk Demo) [!]
	0x07861842A12EBC9Full, // Excitebike 64 (U) [!]
	0x1739EFBAD0B43A68ull, // Kobe Bryant in NBA Courtside (E) [!]
	0x616B84948A509210ull, // Kobe Bryant's NBA Courtside (U) [!]
	0xA197CB527520DE0Eull, // Madden Football 64 (E) [!]
	0x13836389265B3C76ull, // Madden Football 64 (U) [!]
	0xC56741600F5F453Cull, // Mario Party 3 (E) (M4) [!]
	0x0B0AB4CD7B158937ull, // Mario Party 3 (J) [!]
	0x7C3829D96E8247CEull, // Mario Party 3 (U) [!]
	0x839F3AD5406D15FAull, // Mario Tennis (E) [!]
	0x5001CF4FF30CB3BDull, // Mario Tennis (U) [!]
	0x3A6C42B51ACADA1Bull, // Mario Tennis 64 (J) [!]
	0x147E0EDB36C5B12Cull, // Neon Genesis Evangelion (J) [!]
	0xF468118CE32EE44Eull, // PD Ultraman Battle Collection 64 (J) [!]
	0xE4B08007A602FF33ull, // Perfect Dark (E) (M5) [!]
	0x96747EB4104BB243ull, // Perfect Dark (J) [!]
	0xDDF460CC3CA634C0ull, // Perfect Dark (U) (V1.0) [!]
	0x41F2B98FB458B466ull, // Perfect Dark (U) (V1.1) [!]
	0xFEE970104E94A9A0ull, // RR64 - Ridge Racer 64 (E) [!]
	0x2500267E2A7EC3CEull, // RR64 - Ridge Racer 64 (U) [!]
	0x53ED2DC406258002ull, // Star Wars Episode I - Racer (E) (M3) [!]
	0x61F5B152046122ABull, // Star Wars Episode I - Racer (J) [!]
	0x72F703986556A98Bull, // Star Wars Episode I - Racer (U) [!]
	0x2DCFCA608354B147ull, // Yoshi Story (J) [!]
	0xD3F97D496924135Bull, // Yoshi's Story (E) (M3) [!]
};

bool cartEeprom16k(const uint8_t* rom) {
	uint64_t id = (uint64_t) be32(rom + 0x10) << 32 | be32(rom + 0x14);
	for (size_t i = 0; i < sizeof(eeprom16k) / sizeof(*eeprom16k); ++i) {
		if (eeprom16k[i] == id) {
			return true;
		}
	}
	return false;
}
