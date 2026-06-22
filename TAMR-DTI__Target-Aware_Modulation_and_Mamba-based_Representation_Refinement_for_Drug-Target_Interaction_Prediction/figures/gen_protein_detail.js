// DRAFT detail diagram: Protein Representation = ligand-conditioned FiLM fused into 3 ProteinACmix layers.
// Mirrors src/models/models.py ProteinACmix.forward (lines 491-502) exactly.
// Run: node gen_protein_detail.js -> protein_film_detail_draft.svg
const fs = require('fs');
const W = 1500, H = 824;

const C = {
  ink:'#1f2733', sub:'#5a6675',
  navy:'#27408b', red:'#c0392b',
  bP:'#eef2fb', bS:'#c4d3ee', bT:'#2c4a99', bChip:'#5b7fcf',
  ff:'#e4ecf9', ffS:'#c4d3ee',
  ac:'#cdddf4', acS:'#a9c2e8',
  gT:'#22815a', gMod:'#d7ecdf', gS:'#bcd9c8', gChip:'#3fa178',
  film:'#c0392b', filmFill:'#fdecea', filmS:'#eec9c4',
  panel:'#f6f8fc', panelS:'#dde6f4'
};
const P=[]; const out=s=>P.push(s);
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const cxf=b=>b.x+b.w/2, cyf=b=>b.y+b.h/2;

function rr(x,y,w,h,r,fill,stroke,sw=1.4){return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" ry="${r}" fill="${fill}"${stroke?` stroke="${stroke}" stroke-width="${sw}"`:''}/>`;}
function T(x,y,s,{cls='bl',fill=C.ink,anchor='start'}={}){return `<text x="${x}" y="${y}" class="${cls}" fill="${fill}" text-anchor="${anchor}">${s}</text>`;}
function TL(x,y,lines,{cls='bl',fill=C.ink,anchor='middle',lh=18}={}){return lines.map((l,i)=>T(x,y+i*lh,l,{cls,fill,anchor})).join('');}
function sub(m,s){return `${esc(m)}<tspan class="sm" baseline-shift="-18%">${esc(s)}</tspan>`;}
function sup(m,s){return `${esc(m)}<tspan class="sm" baseline-shift="32%">${esc(s)}</tspan>`;}
function chips(x,y,n,fill,{s=14,g=6}={}){let o='',cx=x;for(let i=0;i<n;i++){o+=rr(cx,y,s,s,3,fill,'#fff',1);cx+=s+g;}o+=T(cx+1,y+s-2,'…',{cls:'bl',fill:C.sub});return o;}
function A(pts,{color=C.navy,dash=false,sw=2.3}={}){const d='M '+pts.map(p=>p.join(' ')).join(' L ');const mk=color===C.red?'aR':'aN';return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${sw}"${dash?' stroke-dasharray="8 6"':''} marker-end="url(#${mk})" stroke-linejoin="round" stroke-linecap="round"/>`;}
function filmNode(cx,cy,r){return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="#fff" stroke="${C.film}" stroke-width="2.2"/>`+
  `<line x1="${cx-r*0.5}" y1="${cy-r*0.5}" x2="${cx+r*0.5}" y2="${cy+r*0.5}" stroke="${C.film}" stroke-width="2.2"/>`+
  `<line x1="${cx-r*0.5}" y1="${cy+r*0.5}" x2="${cx+r*0.5}" y2="${cy-r*0.5}" stroke="${C.film}" stroke-width="2.2"/>`;}

// ---- title ----
out(T(40,34,'Protein Representation — ligand-conditioned FiLM fused into 3 ProteinACmix layers',{cls:'h1',fill:C.bT}));
out(T(40,56,'draft for reference · matches ProteinACmix.forward (models.py:491–502)',{cls:'cap',fill:C.sub}));

// ===== input chain =====
const SEQ={x:40,y:78,w:150,h:96};
out(rr(SEQ.x,SEQ.y,SEQ.w,SEQ.h,8,'#fff',C.bS,1.3));
out(TL(cxf(SEQ),SEQ.y+24,['Protein Sequence','/ Structure'],{cls:'bl',fill:C.bT,lh:18}));
out(TL(cxf(SEQ),SEQ.y+62,['MAVSEQLK…','(BioRender ribbon)'],{cls:'mono',fill:C.sub,lh:18}));

const BERT={x:222,y:96,w:108,h:60};
out(rr(BERT.x,BERT.y,BERT.w,BERT.h,10,C.bP,C.bS,1.4));out(T(cxf(BERT),cyf(BERT)+6,'ProtBERT',{cls:'ml',fill:C.bT,anchor:'middle'}));

const P0={x:366,y:96,w:188,h:60};
out(rr(P0.x,P0.y,P0.w,P0.h,9,'#fff',C.bS,1.3));out(T(P0.x+12,P0.y+24,sup('Protein Seed Tokens  P','0'),{cls:'bl',fill:C.bT}));out(chips(P0.x+12,P0.y+h0(P0),3,C.bChip));
function h0(b){return b.h-22;}

const PG={x:366,y:210,w:188,h:58};
out(rr(PG.x,PG.y,PG.w,PG.h,9,'#fff',C.bS,1.3));out(T(PG.x+12,PG.y+22,sub('Protein Seed Context  p','g'),{cls:'bl',fill:C.bT}));out(chips(PG.x+12,PG.y+PG.h-22,3,C.bChip));

// d_g
const DG={x:606,y:92,w:236,h:62};
out(rr(DG.x,DG.y,DG.w,DG.h,9,C.gMod,C.gS,1.4));out(T(cxf(DG),DG.y+26,sub('Ligand Context  d','g'),{cls:'bl',fill:C.gT,anchor:'middle'}));out(T(cxf(DG),DG.y+46,'= mean over fused drug tokens',{cls:'cap',fill:C.sub,anchor:'middle'}));

// ===== 3 FiLM-MLP boxes =====
const blocks=[120,510,900];
const MLP=blocks.map((bx,i)=>({x:bx+150-78,y:250,w:156,h:60,idx:i+1}));
MLP.forEach(m=>{out(rr(m.x,m.y,m.w,m.h,9,C.filmFill,C.filmS,1.3));
  out(T(cxf(m),m.y+22,'FiLM-MLP'+sub('',m.idx),{cls:'bl',fill:C.film,anchor:'middle'}));
  out(T(cxf(m),m.y+40,'Linear→SiLU→Linear',{cls:'cap',fill:C.sub,anchor:'middle'}));});

// ===== 3 ProteinACmix layers =====
const BY=372, BH=176, BW=300;
const layer=blocks.map((bx,i)=>{
  const o={x:bx,y:BY,w:BW,h:BH,idx:i+1};
  out(rr(o.x,o.y,o.w,o.h,12,C.panel,C.panelS,1.6));
  out(T(o.x+16,o.y+24,'ProteinACmix layer '+(i+1),{cls:'cap2',fill:C.bT}));
  const FF={x:o.x+18,y:o.y+50,w:100,h:104};
  out(rr(FF.x,FF.y,FF.w,FF.h,9,C.ff,C.ffS,1.3));out(TL(cxf(FF),FF.y+44,['Feed','Forward'],{cls:'bl',fill:C.bT,lh:18}));out(T(cxf(FF),FF.y+86,'½·FF(x)+½·x',{cls:'cap',fill:C.sub,anchor:'middle'}));
  const fcx=o.x+150, fcy=o.y+102;
  const AC={x:o.x+182,y:o.y+50,w:100,h:104};
  out(rr(AC.x,AC.y,AC.w,AC.h,9,C.ac,C.acS,1.3));out(TL(cxf(AC),AC.y+40,['ACmix'],{cls:'bl',fill:C.bT,lh:18}));out(TL(cxf(AC),AC.y+62,['dynamic conv','+ self-attn'],{cls:'cap',fill:C.sub,lh:15}));
  o.FF=FF;o.AC=AC;o.fcx=fcx;o.fcy=fcy;return o;
});

// Protein Tokens P
const PT={x:1244,y:434,w:180,h:64};
out(rr(PT.x,PT.y,PT.w,PT.h,9,'#fff',C.bS,1.3));out(T(PT.x+12,PT.y+24,'Protein Tokens  P',{cls:'bl',fill:C.bT}));out(chips(PT.x+12,PT.y+PT.h-22,3,C.bChip));

// ===== connectors =====
const E=(b,s)=>s==='r'?[b.x+b.w,cyf(b)]:s==='l'?[b.x,cyf(b)]:s==='t'?[cxf(b),b.y]:[cxf(b),b.y+b.h];
out(A([E(SEQ,'r'),E(BERT,'l')]));
out(A([E(BERT,'r'),E(P0,'l')]));
// P0 -> pg (pool)
out(A([E(P0,'b'),E(PG,'t')]));
out(T(P0.x+P0.w+10,P0.y+P0.h+30,'masked mean-pool',{cls:'cap',fill:C.sub}));
out(T(P0.x+P0.w+10,P0.y+P0.h+46,'(over residues)',{cls:'cap',fill:C.sub}));
// pg -> out to conformer weighting
out(A([E(PG,'r'),[PG.x+PG.w+40,cyf(PG)]]));
out(T(PG.x+PG.w+48,cyf(PG)-4,'→ Target-aware',{cls:'cap',fill:C.sub}));
out(T(PG.x+PG.w+48,cyf(PG)+12,'Conformer Weighting (α_k)',{cls:'cap',fill:C.sub}));
// d_g -> 3 FiLM-MLP (red dashed, fan; bus above the p_g box)
MLP.forEach(m=>out(A([E(DG,'b'),[cxf(DG),174],[cxf(m),174],[cxf(m),m.y]],{color:C.red,dash:true})));
// (gamma,beta) label under each FiLM-MLP
MLP.forEach(m=>out(T(cxf(m),m.y+m.h+18,'(γ'+String.fromCharCode(8320+m.idx)+', β'+String.fromCharCode(8320+m.idx)+')',{cls:'cap',fill:C.film,anchor:'middle'})));
// FiLM-MLP -> FiLM node
layer.forEach((o,i)=>out(A([[cxf(MLP[i]),MLP[i].y+MLP[i].h+24],[o.fcx,o.fcy-17]],{color:C.red,dash:true})));
// internal FF -> FiLM -> ACmix
layer.forEach(o=>{out(filmNode(o.fcx,o.fcy,17));
  out(A([[o.FF.x+o.FF.w,o.fcy],[o.fcx-17,o.fcy]]));
  out(A([[o.fcx+17,o.fcy],[o.AC.x,o.fcy]]));});
// chain: P0 -> L1, L_i -> L_{i+1}, L3 -> P
out(A([E(P0,'l'),[P0.x-30,cyf(P0)],[P0.x-30,352],[layer[0].FF.x+layer[0].FF.w/2,352],[layer[0].FF.x+layer[0].FF.w/2,layer[0].FF.y]]));
out(T(P0.x-150,346,sup('P','0')+' (seed tokens) in',{cls:'cap',fill:C.sub}));
for(let i=0;i<2;i++){out(A([[layer[i].AC.x+layer[i].AC.w,layer[i].fcy],[layer[i+1].FF.x,layer[i+1].fcy]]));}
out(A([[layer[2].AC.x+layer[2].AC.w,layer[2].fcy],E(PT,'l')]));

// ===== formula / legend box =====
const FB={x:120,y:592,w:1304,h:200};
out(rr(FB.x,FB.y,FB.w,FB.h,12,'#fbfcfe',C.panelS,1.4));
out(T(FB.x+24,FB.y+34,'How it fuses (per layer, repeated 3×)',{cls:'h2',fill:C.bT}));
const fy=FB.y+66, lh=30, fx=FB.x+24;
out(T(fx,fy,'1.  half-residual FF:',{cls:'bl',fill:C.ink})+T(fx+200,fy,'h = ½·FFᵢ(x) + ½·x',{cls:'mono2',fill:C.ink}));
out(T(fx,fy+lh,'2.  ligand FiLM:',{cls:'bl',fill:C.film})+T(fx+200,fy+lh,'h ← h · (1 + s·tanh(γᵢ)) + s·βᵢ      (γᵢ, βᵢ) = split( FiLM-MLPᵢ(d_g) ),   s = film_scale = 0.1',{cls:'mono2',fill:C.ink}));
out(T(fx,fy+2*lh,'3.  ACmix block:',{cls:'bl',fill:C.bT})+T(fx+200,fy+2*lh,'x ← ACmixᵢ(h, mask)      (dynamic conv + self-attention)',{cls:'mono2',fill:C.ink}));
out(T(fx,fy+3*lh+6,'• 3 independent FiLM-MLPs → 3 separate (γ,β) pairs, all from the SAME ligand context d_g.   • γ,β are per-channel, broadcast over all residues.',{cls:'cap',fill:C.sub}));
out(T(fx,fy+3*lh+30,'• d_g = mean of fused drug tokens.   • p_g = masked-mean(P⁰), used only to score conformers (α_k) — not part of this stack.',{cls:'cap',fill:C.sub}));

const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="Arial, Helvetica, sans-serif">
<defs>
<marker id="aN" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="9" refY="5" orient="auto"><path d="M0,0.6 L10,5 L0,9.4 Z" fill="${C.navy}"/></marker>
<marker id="aR" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="9" refY="5" orient="auto"><path d="M0,0.6 L10,5 L0,9.4 Z" fill="${C.red}"/></marker>
</defs>
<style>
text{font-family:Arial,Helvetica,sans-serif;}
.h1{font-size:24px;font-weight:700;} .h2{font-size:18px;font-weight:700;} .cap{font-size:13px;} .cap2{font-size:14px;font-weight:700;}
.ml{font-size:18px;font-weight:700;} .bl{font-size:15px;font-weight:700;}
.mono{font-size:12px;font-family:'Consolas','Courier New',monospace;}
.mono2{font-size:14px;font-family:'Consolas','Courier New',monospace;}
.sm{font-size:11px;font-weight:700;}
</style>
<rect width="${W}" height="${H}" fill="#ffffff"/>
${P.join('\n')}
</svg>`;
fs.writeFileSync(__dirname+'/protein_film_detail_draft.svg',svg);
console.log('wrote protein_film_detail_draft.svg',svg.length,'bytes');
