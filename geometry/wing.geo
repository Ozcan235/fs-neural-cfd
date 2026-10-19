// Formula Student ön kanat, yer etkisinde 2D iki elemanlı profil (ana eleman + flap).
// Parametreler komut satırından değiştirilir:
//   gmsh wing.geo -3 -setnumber AoA 6 -setnumber hc 0.15 -setnumber Flap 25 -format msh22 -o wing.msh
// Koordinatlar: akış +x yönünde, zemin y = 0, profil ters çevrilmiş (bastırma kuvveti üretir).
// Açılar saat yönünün tersine pozitif: firar kenarı yukarı = daha fazla bastırma.

SetFactory("Built-in");

DefineConstant[
  AoA    = {4,     Name "Parametreler/01 Hucum acisi [deg]"},
  hc     = {0.20,  Name "Parametreler/02 Yerden yukseklik h/c"},
  Flap   = {20,    Name "Parametreler/03 Flap acisi [deg] (ana elemana gore)"},
  c      = {0.25,  Name "Geometri/Ana eleman veteri [m]"},
  cf     = {0.40,  Name "Geometri/Flap veter orani c_f/c"},
  gap    = {0.020, Name "Geometri/Flap dikey ofseti /c"},
  ovl    = {0.000, Name "Geometri/Flap bindirmesi (overlap) /c"},
  nacaM  = {0.04,  Name "Geometri/Ana eleman NACA m"},
  nacaP  = {0.40,  Name "Geometri/Ana eleman NACA p"},
  nacaT  = {0.12,  Name "Geometri/Ana eleman NACA t"},
  flapM  = {0.04,  Name "Geometri/Flap NACA m"},
  flapP  = {0.40,  Name "Geometri/Flap NACA p"},
  flapT  = {0.12,  Name "Geometri/Flap NACA t"},
  span   = {0.01,  Name "Geometri/Ekstrüzyon kalinligi [m] (Aref = c*span)"},
  Lup    = {5,     Name "Alan/Giris mesafesi /c"},
  Ldown  = {10,    Name "Alan/Cikis mesafesi /c"},
  Htop   = {5,     Name "Alan/Ust sinir yuksekligi /c"},
  Npts   = {120,   Name "Ag/Yuzey basina profil noktasi"},
  yFirst = {2e-5,  Name "Ag/Ilk hucre yuksekligi [m]"},
  blRat  = {1.2,   Name "Ag/Sinir tabakasi buyume orani"},
  blThk  = {1.5e-3,Name "Ag/Sinir tabakasi toplam kalinligi [m]"},
  sWing  = {0.004, Name "Ag/Profil yuzey eleman boyu /c"},
  sWake  = {0.02,  Name "Ag/Iz bolgesi eleman boyu /c"},
  sFar   = {0.5,   Name "Ag/Uzak alan eleman boyu /c"},
  wakeL  = {3,     Name "Ag/Iz bolgesi uzunlugu /c"}
];

ga = AoA*Pi/180;
allPts[] = {};
ymin = 1e9;

// NACA 4 haneli profil, kapalı firar kenarı, kosinüs dağılımı, ters çevrilmiş.
// Girdi: fc, fm, fp, ft, fx0, fy0, frot.  Çıktı: fUp[], fLo[], fLE, fTE
Macro NacaFoil
  fr = frot*Pi/180;
  fUp[] = {}; fLo[] = {};
  For i In {0:Npts}
    b  = Pi*i/Npts;
    xx = 0.5*(1 - Cos(b));
    yt = 5*ft*(0.2969*Sqrt(xx) - 0.1260*xx - 0.3516*xx^2 + 0.2843*xx^3 - 0.1036*xx^4);
    If (fm > 0 && xx < fp)
      yc = fm/fp^2*(2*fp*xx - xx^2);       dy = 2*fm/fp^2*(fp - xx);
    ElseIf (fm > 0)
      yc = fm/(1-fp)^2*((1-2*fp) + 2*fp*xx - xx^2); dy = 2*fm/(1-fp)^2*(fp - xx);
    Else
      yc = 0; dy = 0;
    EndIf
    th = Atan(dy);
    nside = 2;
    If (i == 0 || i == Npts)
      nside = 1;
    EndIf
    For s In {0:nside-1}
      sg = 1 - 2*s;
      px =  (xx - sg*yt*Sin(th))*fc;
      py = -(yc + sg*yt*Cos(th))*fc;
      qx = fx0 + px*Cos(fr) - py*Sin(fr);
      qy = fy0 + px*Sin(fr) + py*Cos(fr);
      gx = qx*Cos(ga) - qy*Sin(ga);
      gy = qx*Sin(ga) + qy*Cos(ga);
      ymin = Min(ymin, gy);
      pt = newp; Point(pt) = {gx, gy, 0, 1};
      allPts[] += pt;
      If (i == 0)
        fLE = pt;
      ElseIf (i == Npts)
        fTE = pt;
      ElseIf (s == 0)
        fUp[] += pt;
      Else
        fLo[] += pt;
      EndIf
    EndFor
  EndFor
Return

// Ana eleman: hücum kenarı orijinde
fc = c; fm = nacaM; fp = nacaP; ft = nacaT; fx0 = 0; fy0 = 0; frot = 0;
Call NacaFoil;
mLE = fLE; mTE = fTE; mUp[] = fUp[]; mLo[] = fLo[];

// Flap: hücum kenarı ana elemanın firar kenarına göre konumlanır (ana eleman ekseninde)
fc = cf*c; fm = flapM; fp = flapP; ft = flapT; fx0 = c*(1 - ovl); fy0 = c*gap; frot = Flap;
Call NacaFoil;
kLE = fLE; kTE = fTE; kUp[] = fUp[]; kLo[] = fLo[];

// En alt nokta zeminden h = hc*c yükseklikte olacak şekilde kaydır
Translate {0, hc*c - ymin, 0} { Point{allPts[]}; }

m1 = newl; Spline(m1) = {mLE, mUp[], mTE};
m2 = newl; Spline(m2) = {mTE, mLo[{#mLo[]-1:0:-1}], mLE};
k1 = newl; Spline(k1) = {kLE, kUp[], kTE};
k2 = newl; Spline(k2) = {kTE, kLo[{#kLo[]-1:0:-1}], kLE};

// Akış alanı
xin = -Lup*c; xout = (1 + Ldown)*c; ytop = Htop*c;
d1 = newp; Point(d1) = {xin,  0,    0, 1};
d2 = newp; Point(d2) = {xout, 0,    0, 1};
d3 = newp; Point(d3) = {xout, ytop, 0, 1};
d4 = newp; Point(d4) = {xin,  ytop, 0, 1};
lg = newl; Line(lg) = {d1, d2};   // zemin
lo = newl; Line(lo) = {d2, d3};   // çıkış
lt = newl; Line(lt) = {d3, d4};   // üst
li = newl; Line(li) = {d4, d1};   // giriş

cOut = newll; Curve Loop(cOut) = {lg, lo, lt, li};
cMain = newll; Curve Loop(cMain) = {m1, m2};
cFlap = newll; Curve Loop(cFlap) = {k1, k2};
srf = news; Plane Surface(srf) = {cOut, cMain, cFlap};

// Ağ boyutu alanları
Field[1] = BoundaryLayer;
Field[1].CurvesList = {m1, m2, k1, k2};
Field[1].FanPointsList = {mTE, kTE};
Field[1].Size = yFirst;
Field[1].Ratio = blRat;
Field[1].Thickness = blThk;
Field[1].Quads = 1;
BoundaryLayer Field = 1;

Field[2] = Distance;
Field[2].CurvesList = {m1, m2, k1, k2};
Field[2].Sampling = 400;

Field[3] = Threshold;
Field[3].InField = 2;
Field[3].SizeMin = sWing*c;
Field[3].SizeMax = sFar*c;
Field[3].DistMin = 0.01*c;
Field[3].DistMax = 2*c;

Field[4] = Box;
Field[4].VIn = sWake*c;
Field[4].VOut = sFar*c;
Field[4].XMin = -0.3*c;
Field[4].XMax = (1 + wakeL)*c;
Field[4].YMin = 0;
Field[4].YMax = hc*c + 0.8*c;
Field[4].Thickness = 0.5*c;

Field[5] = Min;
Field[5].FieldsList = {3, 4};
Background Field = 5;

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.MshFileVersion = 2.2;

// Tek hücre kalınlığında ekstrüzyon (OpenFOAM 2D: ön/arka yüzler "empty")
ex[] = Extrude {0, 0, span} { Surface{srf}; Layers{1}; Recombine; };
// ex[0] = karşı yüz, ex[1] = hacim, ex[2..] = sınır eğrilerinin sırasıyla yan yüzleri
Physical Surface("ground")       = {ex[2]};
Physical Surface("outlet")       = {ex[3]};
Physical Surface("top")          = {ex[4]};
Physical Surface("inlet")        = {ex[5]};
Physical Surface("wing_main")    = {ex[6], ex[7]};
Physical Surface("wing_flap")    = {ex[8], ex[9]};
Physical Surface("frontAndBack") = {srf, ex[0]};
Physical Volume("fluid")         = {ex[1]};
