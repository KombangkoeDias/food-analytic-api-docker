import os

config = {
    "model_name": "convnext_large_384_in22ft1k",
    "image_size": (384, 384),
    "model_weight_filepath": os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_weight.pt"),
    "idx_to_class": {0: 'กรอบเค็ม', 1: 'กระท้อน', 2: 'กระยาสารท', 3: 'กระเพาะปลา', 4: 'กล้วย', 5: 'กล้วยกวน', 6: 'กล้วยตาก', 7: 'กล้วยทอด', 8: 'กล้วยบวชชี', 9: 'กล้วยเชื่อม', 10: 'กั้งสด', 11: 'กาแฟ', 12: 'กีวี่', 13: 'กุนเชียง', 14: 'กุ้งจ่อม', 15: 'กุ้งต้ม', 16: 'กุ้งนึ่ง', 17: 'กุ้งผัดผงกะหรี่', 18: 'ก๋วยจั๊บ', 19: 'ก๋วยจั๊บญวน', 20: 'ก๋วยเตี๋ยวหมู', 21: 'ก๋วยเตี๋ยวเซี่ยงไฮ้', 22: 'ก๋วยเตี๋ยวเนื้อสับ', 23: 'ก๋วยเตี๋ยวเป็ด', 24: 'ก๋วยเตี๋ยวเย็นตาโฟ', 25: 'ก๋วยเตี๋ยวเส้นปลา', 26: 'ก๋วยเตี๋ยวแกง', 27: 'ขนมครก', 28: 'ขนมครกสิงคโปร์', 29: 'ขนมจาก', 30: 'ขนมจีนน้ำยา', 31: 'ขนมจีบ', 32: 'ขนมชั้น', 33: 'ขนมดู', 34: 'ขนมตาล', 35: 'ขนมถ้วยฟู', 36: 'ขนมบดิน', 37: 'ขนมบ้าบิ่น', 38: 'ขนมปัง', 39: 'ขนมปังขาไก่', 40: 'ขนมปังชีสเชคไส้สับปะรด', 41: 'ขนมปังนมสด', 42: 'ขนมปังโฮลวีท', 43: 'ขนมผักกาด', 44: 'ขนมฝักบัว', 45: 'ขนมมาร์เมลโล', 46: 'ขนมลา', 47: 'ขนมเทียน', 48: 'ขนมเบื้อง', 49: 'ขนมเม็ดขนุน', 50: 'ขนมโก๋', 51: 'ขนมใส่ไส้', 52: 'ขนมไหว้พระจันทร์', 53: 'ขนุน', 54: 'ขนุนอบกรอบ', 55: 'ข้อไก่ทอด', 56: 'ข้าวกล้อง', 57: 'ข้าวขาหมู', 58: 'ข้าวคลุกกะปิ', 59: 'ข้าวซอยตัด', 60: 'ข้าวซอยไก่', 61: 'ข้าวตัง', 62: 'ข้าวต้มมัด', 63: 'ข้าวต้มหมู', 64: 'ข้าวผัดหมู', 65: 'ข้าวมันไก่', 66: 'ข้าวมันไก่ทอด', 67: 'ข้าวยำปักษ์ใต้', 68: 'ข้าวหน้าเป็ด', 69: 'ข้าวหมกไก่', 70: 'ข้าวหมาก', 71: 'ข้าวหมูทอด', 72: 'ข้าวหมูแดง', 73: 'ข้าวเกรียบ', 74: 'ข้าวเกรียบปากหม้อไส้หมู', 75: 'ข้าวเหนียวนึ่ง', 76: 'ข้าวเหนียวปิ้ง', 77: 'ข้าวโพดเหลืองต้ม', 78: 'คอหมูย่าง', 79: 'คากิต้มพะโล้', 80: 'คุกกี้ช็อคโกแลต', 81: 'คุกกี้สิงคโปร์', 82: 'คุกกี้เนยสด', 83: 'ชานม', 84: 'ซาลาเปาไส้หมูสับ', 85: 'ซาหริ่ม', 86: 'ซี่โครงหมูต้ม', 87: 'ซุปฟัก', 88: 'ซุปมะเขือ', 89: 'ซุปหน่อไม้', 90: 'ซุปหางวัว', 91: 'ซุปไก่', 92: 'ตะขบ', 93: 'ตะโก้', 94: 'ตับหมูทอด', 95: 'ตั๊กแตนทอด', 96: 'ตำทะเล', 97: 'ตำไทย', 98: 'ต้มข่าไก่', 99: 'ต้มยำกุ้ง', 100: 'ต้มยำไก่', 101: 'ถั่วปากอ้าทอด', 102: 'ถั่วลิสงทอด', 103: 'ถั่วลิสงเคลือบน้ำตาล', 104: 'ถั่วเขียวต้ม', 105: 'ถั่วแระต้ม', 106: 'ทองม้วน', 107: 'ทองหยอด', 108: 'ทองหยิบ', 109: 'ทอดมันกุ้ง', 110: 'ทอดมันปลากราย', 111: 'ทับทิม', 112: 'ทุเรียน', 113: 'ทุเรียนกวน', 114: 'ทุเรียนทอดอบกรอบ', 115: 'นมถั่วเหลือง', 116: 'น้อยหน่า', 117: 'น้ำตาลสด', 118: 'น้ำนมข้าวโพด', 119: 'น้ำพริกอ่อง', 120: 'น้ำมะพร้าว', 121: 'น้ำมะม่วง', 122: 'น้ำมะเขือเทศ', 123: 'น้ำลิ้นจี่', 124: 'น้ำสตรอเบอร์รี่', 125: 'น้ำสับปะรด', 126: 'น้ำส้ม', 127: 'น้ำองุ่น', 128: 'น้ำเก๊กฮวย', 129: 'น้ำเฉาก๊วย', 130: 'น้ำเสาวรส', 131: 'บลูเบอร์รี่', 132: 'บะจ่าง', 133: 'บะหมี่หมูแดง', 134: 'บะหมี่แห้ง', 135: 'บัวลอย', 136: 'บ๊วยดอง', 137: 'ปลาซาบะย่างซีอิ๊ว', 138: 'ปลาดุกทอด', 139: 'ปลาดุกย่าง', 140: 'ปลาดุกแดดเดียว', 141: 'ปลาทูนึ่ง', 142: 'ปลานิลทอด', 143: 'ปลาสลิดทอด', 144: 'ปลาสำลีนึ่ง', 145: 'ปลาส้มทอด', 146: 'ปลาอินทรีย์ทอด', 147: 'ปอเปี๊ยะทอด', 148: 'ปอเปี๊ยะสด', 149: 'ปาท่องโก๋', 150: 'ปีกไก่ทอด', 151: 'ปูม้านึ่ง', 152: 'ผัดกระเพรา', 153: 'ผัดขี้เมา', 154: 'ผัดซีอิ๊วหมู', 155: 'ผัดผักรวมมิตร', 156: 'ผัดหอยลาย', 157: 'ผัดเปรี้ยวหวานไก่', 158: 'ผัดเผ็ดกบ', 159: 'ผัดเผ็ดถั่วฝักยาว', 160: 'ผัดเผ็ดไก่', 161: 'ผัดไทย', 162: 'ฝรั่ง', 163: 'พะแนงไก่หมู', 164: 'พายกรอบ', 165: 'พิชตาชิโอ', 166: 'พิซซ่า', 167: 'ฟองเต้าหู้', 168: 'ฟักทอง', 169: 'ฟักทองฉาบ', 170: 'มะขามคลุกน้ำตาล', 171: 'มะขามป้อม', 172: 'มะขามหวาน', 173: 'มะขามเทศ', 174: 'มะตะบะ', 175: 'มะปราง', 176: 'มะม่วง', 177: 'มะม่วงกวน', 178: 'มะม่วงหิมพานต์', 179: 'มะม่วงอบแห้ง', 180: 'มะม่วงแช่อิ่ม', 181: 'มะยงชิด', 182: 'มะยม', 183: 'มะละกอสุก', 184: 'มะเดื่อสด', 185: 'มะเฟือง', 186: 'มะเม่า', 187: 'มะไฟ', 188: 'มักกะโรนีหมู', 189: 'มังคุด', 190: 'มันฝรั่งทอด', 191: 'มัสมั่นเนื้อ', 192: 'มาม่าต้มยำ', 193: 'ยำรวมมิตรทะเล', 194: 'ระกำ', 195: 'รังนก', 196: 'ราดหน้าหมู', 197: 'ลองกอง', 198: 'ลอดช่อง', 199: 'ละมุด', 200: 'ลางสาด', 201: 'ลาบหมู', 202: 'ลาบเนื้อ', 203: 'ลาบเลือด', 204: 'ลำไย', 205: 'ลำไยอบแห้ง', 206: 'ลิ้นจี่', 207: 'ลูกชิ้น', 208: 'ลูกชุบ', 209: 'ลูกตะลิงปลิง', 210: 'ลูกตาลเชื่อม', 211: 'ลูกหว้า', 212: 'ลูกเกด', 213: 'ลูกเดือยน้ำกะทิ', 214: 'ลูกไหน', 215: 'วาฟเฟิล', 216: 'วุ้นมะพร้าว', 217: 'สตรอเบอร์รี่', 218: 'สตรอเบอร์รี่อบแห้ง', 219: 'สตู', 220: 'สปาเก็ตตี้ซอส', 221: 'สละ', 222: 'สลัดผัก', 223: 'สังขยาฟักทอง', 224: 'สับปะรด', 225: 'สับปะรดกวน', 226: 'สับปะรดอบแห้ง', 227: 'สาคูถั่วดำ', 228: 'สาลี่', 229: 'สเต็กปลาทูน่า', 230: 'ส้ม', 231: 'ส้มฟัก', 232: 'ส้มอบแห้ง', 233: 'ส้มโอ', 234: 'หนอนไม้ไผ่ทอด', 235: 'หนังไก่ทอด', 236: 'หมี่กรอบ', 237: 'หมี่กะทิ', 238: 'หมี่ซั่ว', 239: 'หมี่โคราช', 240: 'หมูกรอบ', 241: 'หมูทอด', 242: 'หมูทุบ', 243: 'หมูสวรรค์', 244: 'หมูสะเต๊ะ', 245: 'หมูสามชั้นทอด', 246: 'หมูหยอง', 247: 'หมูแดง', 248: 'หมูแดดเดียว', 249: 'หมูแผ่น', 250: 'หม่ำหมู', 251: 'หม้อแกง', 252: 'หอยทอด', 253: 'หอยนางรมทรงเครื่อง', 254: 'หอยหลอดแห้ง', 255: 'หอยเสียบดอง', 256: 'หอยแครงลวก', 257: 'หอยแมลงภู่นึ่ง', 258: 'หูฉลาม', 259: 'องุ่นเขียว', 260: 'องุ่นแดง', 261: 'อัลมอนด์', 262: 'ฮอทดอก', 263: 'ฮ่อยจ๊อ', 264: 'เกาลัด', 265: 'เกี๊ยวกุ้ง', 266: 'เกี๊ยวทอด', 267: 'เคบับ', 268: 'เค้กชิฟฟอนกาแฟใบเตย', 269: 'เค้กเนยสด', 270: 'เงาะ', 271: 'เฉาก๊วย', 272: 'เชอร์รี่', 273: 'เต้าคั่ว', 274: 'เต้าส่วน', 275: 'เต้าหู้ทอด', 276: 'เต้าฮวย', 277: 'เนื้อเค็ม-เนื้อแดดเดียว', 278: 'เบอร์เกอร์เนื้อ', 279: 'เป็ดย่าง', 280: 'เผือกกวน', 281: 'เผือกฉาบ', 282: 'เผือกทอด', 283: 'เมล่อน-แคนตาลูป', 284: 'เมี่ยงคำใต้', 285: 'เมี่ยงปลานิลเผา', 286: 'เสาวรส', 287: 'แกงกะทิหมู', 288: 'แกงกะทิไก่', 289: 'แกงกะหรี่', 290: 'แกงขนุน', 291: 'แกงขี้เหล็ก', 292: 'แกงคั่วหอยขม', 293: 'แกงบอน', 294: 'แกงส้มกุ้งชะอม', 295: 'แกงหมูชะมวง', 296: 'แกงอ่อมปลา', 297: 'แกงอ่อมเนื้อ', 298: 'แกงฮังเล', 299: 'แกงเหลืองปลา', 300: 'แกงเหลืองไก่', 301: 'แกงไตปลา', 302: 'แก้วมังกร', 303: 'แซนวิชไก่', 304: 'แตงโม', 305: 'แมคคาเดเมีย', 306: 'แหนมหมู', 307: 'แหนมเนื้อ', 308: 'แห้ว', 309: 'แอปปริคอท', 310: 'แอปเปิ้ลเขียว', 311: 'แอปเปิ้ลแดง', 312: 'โจ๊ก', 313: 'โตเกียวไส้หมูสับ', 314: 'ไก่ชุบแป้งทอด', 315: 'ไก่ต้ม-ไก่ตุ๋น', 316: 'ไก่หยอง', 317: 'ไก่อบซอส', 318: 'ไก่เทอริยากิ', 319: 'ไข่ตุ๋น', 320: 'ไข่ต้ม', 321: 'ไข่นกกระทา', 322: 'ไข่ปลาหมึกทอด', 323: 'ไข่พะโล้', 324: 'ไข่มดแดง', 325: 'ไข่เค็ม', 326: 'ไข่เจียว', 327: 'ไส้กรอกอีสาน', 328: 'ไส้อั่ว', 329: 'ไส้เป็ดต้มพะโล้'},
    "top_k": 3,
}