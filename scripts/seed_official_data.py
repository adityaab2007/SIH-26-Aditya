from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SOURCE_DASHBOARD = "https://ipm.mospi.gov.in/Home/PublicDashboard"
SOURCE_HIGH_VALUE = "https://ipm.mospi.gov.in/Home/GetHighlyValue"
SOURCE_FEB = "https://uatipm.mospi.gov.in/"
SOURCE_MARCH = "https://ipm.mospi.gov.in/Content/PDF/FlashReport_March_2026.pdf"

rows = []
def add(sector, ministry, code, name, original, revised, expenditure, original_end, revised_end, physical=None):
    rows.append({
        "snapshot_date":"2026-05-31",
        "sector":sector,
        "ministry":ministry,
        "project_code":str(code),
        "project_name":name,
        "original_cost_cr":original,
        "revised_cost_cr":revised,
        "expenditure_cr":expenditure,
        "original_end_date":original_end,
        "revised_end_date":revised_end,
        "physical_progress_pct":physical,
        "source_url":SOURCE_DASHBOARD,
    })

# Official PAIMANA public dashboard subset, snapshot May 2026.
# Values below are transcribed from the public project table surfaced by MoSPI.
M_DHE = "Department of Higher Education"
add("Education",M_DHE,602098,"Construction of Permanent Campus for IIM Jammu Phase-1 at Jagti, Nagrota, Jammu, J&K",425,500.91,484.52,"2023-06-30","2025-11-30")
add("Education",M_DHE,602099,"Establishment of Permanent campus Phase 1 of IIM Sirmaur",393,454.78,337.67,"2023-10-31","2026-06-30")
add("Education",M_DHE,605292,"Construction of Permanent Campus of Indian Institute of Management [IIM] Amritsar, Punjab, Phase-I",348,None,293.10,"2023-06-30","2026-11-20")
add("Education",M_DHE,606013,"Additional Campus of NITPatna",499,None,449.96,"2024-07-31","2026-01-31")
add("Education",M_DHE,606112,"Tezpur University Projects",154,None,137.84,"2023-11-08","2025-12-31")
add("Education",M_DHE,606431,"CONSTRUCTION OF PERMANENT CAMPUS FOR IIT PALAKKAD UNDER PHASE 1A",1006,1217.4,686.2,"2022-06-07","2026-06-30")
add("Education",M_DHE,606479,"Banaras Hindu University [Kayakalp of Institute of Medical Sciences]",246,None,209.68,"2022-09-30","2026-03-31")
add("Education",M_DHE,609019,"HEFA projects on development of student hostels, playgrounds, departmental buildings, class room complex and other campus facilities in IIT Kharagpur",500,442,416.2,"2023-03-31","2026-03-31")
add("Education",M_DHE,609736,"Construction of Sikkim University Campus Phase-I Package-II at Yangang, South Sikkim",314,None,292,"2025-01-04","2026-03-31")
add("Education",M_DHE,609996,"Construction of Precast housing and hostels at IIT Hyderabad",262,None,216.1,"2024-10-23","2026-01-01")
add("Education",M_DHE,610315,"IISER BERHAMPUR",1229,843.09,874.09,"2021-03-31","2026-09-30")
add("Education",M_DHE,611957,"Phase III A works at IIT Indore",496,None,78.35,"2027-02-19",None)
add("Education",M_DHE,611961,"Construction of IIT Bhubaneswar -Phase III",454,None,101.36,"2026-11-20",None)
add("Education",M_DHE,611963,"Construction & Development of Infrastructural Facilities in the main campus of NIT Uttarakhand at Sumari, Pauri Garhwal",651,None,347.26,"2027-09-30",None)
add("Education",M_DHE,612128,"Hostel blocks & dining hall block",167,None,122.26,"2026-03-26","2026-07-31")
add("Education",M_DHE,612851,"Phase 1B - Construction at National Institute of Technology, Delhi Campus",333,336.42,327.26,"2025-09-30","2026-01-31")
add("Education",M_DHE,612876,"Construction of Permanent Campus for NIT Mizoram at Lengpui, Aizawl",474,474.23,167.55,"2029-12-31","2022-03-31")
add("Education",M_DHE,612877,"Establishment of NIT Puducherry Permanent Campus [Phase-I and Phase-II]",618,544.34,543.74,"2024-12-31","2026-05-31")
add("Education",M_DHE,612883,"Establishment of Permanent Campus of NIT Sikkim at Khamdong-Gangtok District",686,610.76,283.56,"2028-02-12","2026-12-31")
add("Education",M_DHE,612885,"Construction and Development of National Institute of Technology Meghalaya Permanent Campus at Sohra",495,429.7,370.98,"2022-03-31","2026-10-31")
add("Education",M_DHE,612894,"Construction of Permanent Campus of Central University of Himachal Pradesh at Dehra",252,300.45,223.21,"2024-11-18","2026-03-25")
add("Education",M_DHE,612898,"Construction of Permanent Campus of National Institute of Technology Arunachal Pradesh",593,None,551.3,"2026-08-31","2024-08-31")
add("Education",M_DHE,612916,"Construction of permanent campus of Central University of Gujarat at Village Kundhela",491,None,337.74,"2025-03-31","2026-03-31")
add("Education",M_DHE,613008,"Construction of Permanent Campus of Central Tribal University of Andhra Pradesh Phase-1",305,None,67.73,"2027-06-26",None)
add("Education",M_DHE,613013,"Establishment of NIT Nagaland",276,None,263.25,"2025-12-31",None)
add("Education",M_DHE,616886,"Delhi University Eastern Campus at Surajmal Vihar",373,None,30.13,"2026-10-31",None)
add("Education",M_DHE,617221,"Construction of Girls Hostel, Boys Hostel, Academic Block and allied facilities",211,None,87.16,"2027-10-03",None)
add("Education",M_DHE,617877,"Phase B Construction of Permanent Campus of IIT Bhilai",1189,None,0,"2028-10-31",None)
add("Education",M_DHE,617878,"Phase B of Construction of Permanent Campus of IIT Palakkad",1527,None,9.8,"2028-10-31",None,0)
add("Education",M_DHE,619033,"1000 seater Boys hostel, 500 seater Girls hostel, STP and Faculty Residence",199,None,42.55,"2026-06-30",None)
add("Education","Department of Sports",400081,"National Sports University [NSU], Imphal [Manipur]",906,611.74,575.68,"2024-08-26","2026-08-08")

M_DOT = "Department of Telecommunications"
add("Telecommunication",M_DOT,400013,"Saturation 4G mobile coverage of Uncovered Villages through USOF",26316,15392.8,9791.8,"2024-06-20","2027-03-31")
add("Telecommunication",M_DOT,617413,"Mobile connectivity in Left Wing Extremism [LWE] affected areas-Phase 1 Upgradation",1885,None,376.8,"2025-05-08","2026-03-31")
add("Telecommunication",M_DOT,706775,"BharatNet",61109,188000,46432,"2026-03-31","2027-03-31",82)

M_CIV = "Ministry of Civil Aviation"
add("Aviation & Aviation Infrastructure",M_CIV,701101,"Construction of New Domestic Terminal Building [Phase-I and II] and allied structures at JPNI Airport, Patna",1217,1216.9,1201.92,"2026-05-31","2026-08-31")
add("Aviation & Aviation Infrastructure",M_CIV,701105,"Goa Airport Terminal Building Extension Project",256,255.69,158.78,"2021-12-09","2026-04-30")
add("Aviation & Aviation Infrastructure",M_CIV,701107,"Construction of New Integrated Terminal Building and associated works at Vijayawada Airport",612,611.8,523.14,"2022-09-04","2026-10-31")
add("Aviation & Aviation Infrastructure",M_CIV,701113,"Development of Lal Bahadur Shastri International airport, Varanasi",2870,None,626.15,"2027-07-20","2027-07-20",27)
add("Aviation & Aviation Infrastructure",M_CIV,701126,"Development of Dholera International Greenfield Airport, Gujarat",1305,1551,922.78,"2026-06-30","2026-09-30")
add("Aviation & Aviation Infrastructure",M_CIV,701127,"Construction of New Passenger Terminal Building for Domestic Operations at Jodhpur Airport",480,480,345.45,"2026-03-31","2026-06-15")
add("Aviation & Aviation Infrastructure",M_CIV,701128,"Construction of New Integrated Passenger Terminal Building at Udaipur Airport",887,887,538,"2026-03-10","2026-09-26")
add("Aviation & Aviation Infrastructure",M_CIV,706718,"C/o NITB Imphal Airport",499,499,206.61,"2024-07-13","2026-12-31")
add("Aviation & Aviation Infrastructure",M_CIV,706724,"Guwahati Airport New Integrated Terminal Building Construction Project",1712,2520,2639.99,"2025-03-31","2026-06-30")

M_HFW = "Ministry of Health & Family Welfare"
for code,name,orig,rev,exp,oe,re in [
(707020,"Establishment of new GMC Jamui",250,500,103.21,"2025-10-06",None),
(707021,"Establishment of new GMC Koderma",250,504.8,271,"2026-08-19",None),
(707022,"Establishment of new GMC Chaibasa [Singhbhum]",250,454.1,65,"2025-09-09",None),
(707039,"Establishment of new GMC Gangtok",250,635.21,667.21,"2026-12-31",None),
(707044,"Establishment of new GMC Korba",325,322.85,178.49,"2026-12-31",None),
(707045,"Establishment of new GMC Mahasamund",325,None,322.5,"2026-05-31",None),
(707046,"Establishment of new GMC Kanker",325,325,36,"2027-03-31",None),
(707049,"Establishment of new GMC Panchmahal",325,663.5,457.34,"2026-03-31",None),
(707050,"Establishment of new GMC Porbandar",325,618.98,936.64,"2024-09-30",None),
]: add("Healthcare",M_HFW,code,name,orig,rev,exp,oe,re)

M_HUA = "Ministry of Housing & Urban Affairs"
add("Urban Public Transport",M_HUA,702628,"Kanpur Metro Rail Project",11076,None,9121.15,"2024-05-26","2027-03-31")
add("Urban Public Transport",M_HUA,702629,"Agra Metro Rail Project",8380,None,6102.28,"2024-05-26","2027-06-30")
add("Urban Public Transport",M_HUA,702630,"Surat Metro Rail Project",12020,None,9031.25,"2024-03-09","2027-03-31")
add("Urban Public Transport",M_HUA,702632,"DMRTS Phase - IV [3 Priority Corridors]",24949,None,20545.9,"2025-03-31","2026-12-31")
add("Urban Public Transport",M_HUA,702635,"Construction of Bangalore Metro Rail Project Phase 2",26405,30695.1,29813,"2021-03-22","2026-09-30")
add("Urban Public Transport",M_HUA,702637,"Mumbai Metro Line 3",23136,37276,33648.3,"2023-03-31","2025-08-31")
add("Urban Public Transport",M_HUA,702658,"Nagpur Metro Rail Phase II Development Project",6708,None,1940.06,"2027-12-31",None)
add("Urban Public Transport",M_HUA,702668,"Chennai Metro Rail Phase-II Development Project",63246,None,32503.4,"2029-08-31",None,53)

for code,name,orig,rev,exp,oe,re in [
(702793,"Indore Water Supply Infrastructure Project",287,None,252.5,"2023-03-31","2026-06-30"),
(702808,"Satna Sewerage Management and Treatment Infrastructure Project",192,215.89,200.32,"2024-03-31","2026-08-31"),
(702810,"Singrauli Sewerage Management and Treatment Infrastructure Project",402,438.1,371.17,"2023-03-31","2026-06-30"),
(702833,"Kalyan-Dombivali Water Supply Infrastructure Project",194,None,136.56,"2023-10-31","2026-08-31"),
(702851,"Pune Water Supply Infrastructure Project",236,None,220.77,"2023-03-31","2026-08-31"),
(702949,"Jhansi Water Supply Infrastructure Phase-II Project",562,None,506.52,"2023-07-31","2026-08-31"),
(703021,"Jalandhar Water Supply Infrastructure Project",259,266.64,161.02,"2023-08-28","2026-06-30"),
(703057,"Patiala Water Supply Infrastructure Part-2 Project",287,None,213.64,"2023-10-31","2026-05-31"),
]: add("Waste & Water",M_HUA,code,name,orig,rev,exp,oe,re)

M_PWR = "Ministry of Power"
for code,name,orig,rev,exp,oe,re in [
(602961,"Talcher Thermal Power Station Stage III [2 x 660 MW] Expansion",11844,12543,6644.65,"2027-05-31","2028-05-31"),
(605155,"Kwar HE Project",4526,None,1616.87,"2026-11-10","2028-03-31"),
(605156,"Rangit-IV HE Project [120 MW]",938,1828.11,1672.37,"2024-05-29","2026-11-30"),
(611509,"Koderma Thermal Power Station Phase II [2X800 MW]",14358,None,1964.6,"2029-03-14",None),
(611586,"Sunni Dam Hydro Electric Project [382 MW]",2615,4925.87,1097.78,"2028-04-03","2029-12-31"),
(611698,"240MW Heo HE Project",1939,None,105.5,"2029-07-29",None),
(611699,"186 MW Tato-I H E Project",1750,None,74.83,"2029-07-29",None),
(611716,"700 MW Tato-II HE Project",8146,None,244.35,"2031-09-03",None),
(611930,"Lara Super Thermal Power Project Stage-II [2 x 800 MW]",15530,16106,5005.4,"2028-06-30","2028-11-30"),
]: add("Electricity Generation",M_PWR,code,name,orig,rev,exp,oe,re)

M_RAIL = "Ministry of Railways"
for code,name,orig,rev,exp,oe,re,phys in [
(705394,"Tetlia-Byrnihat New Railway Line",1305,1304.62,1258.8,"2011-11-30","2027-06-30",None),
(705396,"Byrnihat-Shillong New Rail Line- 108 km",8324,8342.28,266.12,"2021-03-31","2027-03-31",None),
(705401,"Murkongselek-Pasighat",435,1250,867.76,"2025-03-31","2026-12-31",None),
(705410,"Nangal Dam-Talwara New Broad Gauge Line",2018,2018,2398.08,"2025-12-31","2027-12-31",None),
(705419,"Gwalior - Seopurkala with extension to Kota [Gauge Conversion]",2913,2913,2245.43,"2025-03-31","2026-02-28",None),
(705424,"Chandigarh-Baddi New Rail Line- 36 km",1540,1540.13,606.07,"2026-12-31","2026-12-31",None),
(705428,"Bhanupalli-Bilaspur-Beri New Railway Line Project",13447,13447,7688.76,"2026-12-31","2027-12-30",None),
(705728,"Mumbai-Ahmedabad High Speed Rail Project- 508 km",108000,108000,90396,"2029-12-31","2029-12-31",60),
]: add("Railways",M_RAIL,code,name,orig,rev,exp,oe,re,phys)

M_PNG = "Ministry of Petroleum & Natural Gas"
for code,name,orig,rev,exp,oe,re in [
(709847,"Pipeline Network Project Rudrasagar",481,481.03,235.68,"2026-04-27","2027-04-30"),
(709848,"MHN TCPP PGC BGC Project [MTPBP]",2329,2354.23,1923.61,"2026-05-19","2026-12-31"),
(709849,"PRP-VIII Group B [NH & BS Asset]",2380,3165.53,2923.49,"2026-02-28","2026-12-31"),
(709850,"Borholla GGS Life Extension Project [BLEP]",199,198.7,103.56,"2026-05-16","2026-09-30"),
(709851,"Integrated Development of 4 Contract Areas under DSF-II",6184,6183.79,1722,"2027-04-30","2027-04-30"),
(709864,"DCU Revamp Project",370,370,299.25,"2024-07-17","2027-03-31"),
(709865,"Power Recovery Turbine [PRT] in PFCCU unit as Energy Conservation Scheme",294,293.5,81.84,"2024-12-31","2026-12-31"),
(701263,"Rajasthan Refinery Project",43129,79459,69997,"2022-10-31","2026-06-30"),
]: add("Oil & Gas",M_PNG,code,name,orig,rev,exp,oe,re,92 if code==701263 else None)

M_PORT = "Ministry of Ports, Shipping and Waterways"
add("Inland Waterways",M_PORT,701565,"Jal Marg Vikas Project",5369,5061.15,4028.82,"2023-12-31","2025-12-31",81)
add("Shipping",M_PORT,400382,"Rewas Port Project",6000,None,5999.89,"2023-03-31",None,88)
add("Shipping",M_PORT,609424,"Development of Container Terminal at Tuna-Tekra, Deendayal Port",4244,None,2385,"2027-03-13","2027-09-30")

# High-value endpoint rows without surfaced official project codes are intentionally
# excluded from the training dataset. We do not invent identifiers for official records.

fields = list(rows[0].keys())
with (RAW / "paimana_projects_may_2026.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)

# Historical high-value snapshots from official PAIMANA/Flash Report surfaces.
history = [
    # snapshot, code, name, sector, original, revised, expenditure, progress, revised_completion, source
    ("2026-02-28","705728","Mumbai-Ahmedabad High Speed Rail Project- 508 km","Railways",108000,108000,None,59,"2029-12-31",SOURCE_FEB),
    ("2026-03-31","705728","Mumbai-Ahmedabad High Speed Rail Project- 508 km","Railways",108000,108000,90396,59.1,"2029-12-31",SOURCE_MARCH),
    ("2026-05-31","705728","Mumbai-Ahmedabad High Speed Rail Project- 508 km","Railways",108000,108000,None,60,"2029-12-31",SOURCE_HIGH_VALUE),
    ("2026-02-28","702668","Chennai Metro Rail Phase-II Development Project","Urban Public Transport",63246,63246,None,52,"2029-08-31",SOURCE_FEB),
    ("2026-03-31","702668","Chennai Metro Rail Phase-II Development Project","Urban Public Transport",63246,63246,32503.4,52,"2029-08-31",SOURCE_MARCH),
    ("2026-05-31","702668","Chennai Metro Rail Phase-II Development Project","Urban Public Transport",63246,63246,None,53,"2029-08-31",SOURCE_HIGH_VALUE),
    ("2026-02-28","706775","BharatNet","Telecommunication",61109,188000,None,82,"2027-03-31",SOURCE_FEB),
    ("2026-03-31","706775","BharatNet","Telecommunication",61109,188000,46432,81.5,"2027-03-31",SOURCE_MARCH),
    ("2026-05-31","706775","BharatNet","Telecommunication",61109,188000,None,82,"2027-03-31",SOURCE_HIGH_VALUE),
    ("2026-02-28","701263","Rajasthan Refinery Project","Oil & Gas",43129,79459,64516,92,"2026-06-30",SOURCE_FEB),
    ("2026-03-31","701263","Rajasthan Refinery Project","Oil & Gas",43129,79459,68837.7,91.6,"2026-06-30",SOURCE_MARCH),
    ("2026-05-31","701263","Rajasthan Refinery Project","Oil & Gas",43129,79459,69997,92,"2026-06-30",SOURCE_HIGH_VALUE),
    ("2026-02-28","701565","Jal Marg Vikas Project","Inland Waterways",5369,5061.15,None,81,"2025-12-31",SOURCE_FEB),
    ("2026-05-31","701565","Jal Marg Vikas Project","Inland Waterways",5369,5061.15,4028.82,81,"2025-12-31",SOURCE_HIGH_VALUE),
]
with (RAW / "paimana_high_value_history.csv").open("w", newline="", encoding="utf-8") as f:
    fields2=["snapshot_date","project_code","project_name","sector","original_cost_cr","revised_cost_cr","expenditure_cr","physical_progress_pct","revised_completion_date","source_url"]
    w=csv.DictWriter(f, fieldnames=fields2); w.writeheader()
    for x in history:
        w.writerow(dict(zip(fields2,x)))

print(f"Wrote {len(rows)} official current project rows and {len(history)} historical snapshots")
