// Generator for the TAMR-DTI architecture figure (publication SVG).
// Run: node gen_arch_svg.js  ->  tamr_dti_architecture_v8.svg
// Design preserves the pptx panel/color language but fixes:
//   (1) connector crowding -> one small consistent arrowhead + routed segments with breathing room
//   (2) panel titles inset INSIDE the panel (never on the rounded border)
//   (3) Arial type scale, top-venue style
const fs = require('fs');

const W = 1960, H = 1120;

// ---- palette (slightly desaturated vs pptx, same hues) ----
const C = {
  ink: '#1f2733', sub: '#5a6675', white: '#ffffff',
  navy: '#27408b',                 // data flow
  red: '#c0392b',                  // target-aware modulation
  purple: '#6d44a6',               // mamba path
  // drug (green)
  gP:'#eef6f1', gS:'#bcd9c8', gT:'#22815a', gMod:'#d4ebde', gChip:'#3fa178',
  // conformer (orange)
  oP:'#fdf3e4', oS:'#f1d6a6', oT:'#c07d12', oMod:'#f7e2bf', oChip:'#e6a23c',
  // fusion (teal-green neutral)
  fP:'#eef6f2', fS:'#c2ddd0', fT:'#2a7d59',
  // protein (blue)
  bP:'#eef2fb', bS:'#c4d3ee', bT:'#2c4a99', bMod:'#d8e2f6', bChip:'#5b7fcf',
  // mamba (purple)
  pP:'#f4eefb', pS:'#d9c8ee', pT:'#6d44a6', pMod:'#e6d8f5', pChip:'#9277c4',
  // biintention (light blue)
  iP:'#eaf1fc', iS:'#cbdaf2', iT:'#2c4a99', iMod:'#d8e2f6',
  // prediction (red)
  rP:'#fdeeeb', rS:'#f1cdc5', rT:'#b23a2c', rMod:'#f6d8d1',
  // badge
  badge:'#f4c534', badgeT:'#5b4a13', star:'#e39a00',
  chipStroke:'#ffffff'
};

const B = {};                       // geometry registry
const reg = (name,x,y,w,h)=>{ B[name]={x,y,w,h}; return B[name]; };
const cx = b => b.x+b.w/2, cy = b => b.y+b.h/2;
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// ---------- primitives ----------
function rrect(x,y,w,h,r,fill,stroke,sw=1.4,extra=''){
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" ry="${r}" fill="${fill}"`+
         (stroke?` stroke="${stroke}" stroke-width="${sw}"`:'')+` ${extra}/>`;
}
function T(x,y,s,{cls='bl',fill=C.ink,anchor='start'}={}){
  return `<text x="${x}" y="${y}" class="${cls}" fill="${fill}" text-anchor="${anchor}">${s}</text>`;
}
// multi-line centered/anchored text
function TL(x,y,lines,{cls='bl',fill=C.ink,anchor='middle',lh=20}={}){
  return lines.map((ln,i)=>T(x,y+i*lh,ln,{cls,fill,anchor})).join('');
}
function sub(main,s){ return `${esc(main)}<tspan class="subsc" baseline-shift="-18%">${esc(s)}</tspan>`; }

// chips: n rounded squares + ellipsis, returns nothing (draws)
function chips(x,y,n,fill,{s=15,g=7}={}){
  let out=''; let cxp=x;
  for(let i=0;i<n;i++){ out+=rrect(cxp,y,s,s,3,fill,C.chipStroke,1); cxp+=s+g; }
  out+=T(cxp+1,y+s-2,'…',{cls:'bl',fill:C.sub});
  return out;
}

// arrow through orthogonal points
function A(pts,{color=C.navy,dash=false,sw=2.4}={}){
  const d='M '+pts.map(p=>p.join(' ')).join(' L ');
  const mk = color===C.red?'aR':color===C.purple?'aP':'aN';
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${sw}"`+
         (dash?' stroke-dasharray="9 7"':'')+` marker-end="url(#${mk})" stroke-linejoin="round" stroke-linecap="round"/>`;
}
// operator circle with a drawn glyph (mult / odot / sigmoid / equals)
function opCircle(x,y,r,kind,color){
  let g='';
  if(kind==='x'){ const k=r*0.5; g=`<line x1="${x-k}" y1="${y-k}" x2="${x+k}" y2="${y+k}" stroke="${color}" stroke-width="2.4"/><line x1="${x-k}" y1="${y+k}" x2="${x+k}" y2="${y-k}" stroke="${color}" stroke-width="2.4"/>`; }
  else if(kind==='eq'){ const k=r*0.46; g=`<line x1="${x-k}" y1="${y-k*0.5}" x2="${x+k}" y2="${y-k*0.5}" stroke="${color}" stroke-width="2.4"/><line x1="${x-k}" y1="${y+k*0.5}" x2="${x+k}" y2="${y+k*0.5}" stroke="${color}" stroke-width="2.4"/>`; }
  else if(kind==='odot'){ g=`<circle cx="${x}" cy="${y+1}" r="${r*0.22}" fill="${color}"/><text x="${x}" y="${y-2}" class="an" fill="${color}" text-anchor="middle" font-weight="700">g</text>`; }
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="#fff" stroke="${color}" stroke-width="2.2"/>${g}`;
}

// ===================================================================
// LAYOUT (geometry only)
// ===================================================================
// panels
reg('PN_DRUG', 36,28,724,330);
reg('PN_CONF', 36,378,724,234);
reg('PN_PROT', 36,632,1050,398);
reg('PN_FUSE', 786,28,300,576);
reg('PN_MAMBA',1112,28,812,250);
reg('PN_BIINT',1112,298,812,430);
reg('PN_PRED', 1112,748,812,282);

// --- drug rows ---
const dInX=56,dInW=148, dEnX=250,dEnW=120, dOuX=448,dOuW=296;
const dR=[123,209,295];                     // row centers
['SMILES','GRAPH','CONF'].forEach((k,i)=>{ reg('IN_'+k,dInX,dR[i]-35,dInW,70); });
['CHEM','GCN','EGNN'].forEach((k,i)=>{ reg('EN_'+k,dEnX,dR[i]-35,dEnW,70); });
['SEM','TOPO','CSUM'].forEach((k,i)=>{ reg('OU_'+k,dOuX,dR[i]-35,dOuW,70); });

// --- conformer ---
reg('CF_PSC', 56,452,156,62);
reg('CF_EN',  56,528,156,62);
reg('CF_MLP', 252,470,104,104);
reg('CF_GEO', 600,470,150,104);
B.CF_OPx=552; B.CF_OPy=522; B.CF_OPr=20;
// alpha bars
B.AB=[{x:392,y:474,w:40,v:'0.23'},{x:392,y:512,w:64,v:'0.47'},{x:392,y:550,w:22,v:'0.08'}];

// --- fusion ---
const fX=806,fW=260;
reg('FU_1D',  fX,108,fW,62);
reg('FU_GEO', fX,196,fW,62);
reg('FU_SIG', fX,286,fW,92);
B.FU_MGx=936; B.FU_MGy=414; B.FU_MGr=17;
reg('FU_FUSED',fX,448,fW,62);
reg('FU_LIG', fX,534,fW,62);

// --- protein ---
reg('PR_SEQ', 56,708,148,300);
reg('PR_BERT',240,818,98,92);
reg('PR_P0',  380,712,150,82);
reg('PR_PG',  380,896,150,82);
reg('PR_FILM',566,800,140,118);
reg('PR_ACMIX',730,800,176,118);
reg('PR_TOK', 936,822,148,80);

// --- mamba ---
reg('MB_IN',  1140,118,170,82);             // Protein Tokens P (in)
reg('MB_SSM', 1340,86,260,128);
B.MB_Gx=1668; B.MB_Gy=150; B.MB_Gr=22;
reg('MB_REF', 1734,108,168,84);

// --- biintention ---
reg('BI_D2P', 1140,360,330,86);
reg('BI_P2D', 1500,360,330,86);
reg('BI_MAP', 1140,470,300,170);
reg('BI_POOL',1500,486,300,120);
reg('BI_JOINT',1148,664,680,52);

// --- prediction ---
reg('PD_MLP', 1150,812,210,150);
reg('PD_SIG', 1430,812,200,150);
reg('PD_OUT', 1700,812,200,150);

// ===================================================================
// EMIT
// ===================================================================
const bg=[], mid=[], fg=[];

// ---- panels (with inset titles) ----
function panel(name,fill,stroke,title,tcolor){
  const b=B[name];
  bg.push(rrect(b.x,b.y,b.w,b.h,18,fill,stroke,1.6));
  if(title) fg.push(T(b.x+26,b.y+38,esc(title),{cls:'pt',fill:tcolor}));
}
panel('PN_DRUG', C.gP,C.gS,'Drug Representation',C.gT);
panel('PN_CONF', C.oP,C.oS,'Target-aware Conformer Weighting',C.oT);
panel('PN_PROT', C.bP,C.bS,'Protein Representation',C.bT);
panel('PN_FUSE', C.fP,C.fS,'Token-wise Fusion Gate',C.fT);
panel('PN_MAMBA',C.pP,C.pS,'Protein-Mamba BiIntention Fusion',C.pT);
panel('PN_BIINT',C.iP,C.iS,'BiIntention · Bidirectional Cross-Attention',C.iT);
panel('PN_PRED', C.rP,C.rS,'Interaction Prediction Head',C.rT);

// ---- module box (solid tint, centered label) ----
function mod(name,fill,stroke,lines,tcolor,{cls='ml',lh=22,subLines=[],subColor}={}){
  const b=B[name];
  fg.push(rrect(b.x,b.y,b.w,b.h,12,fill,stroke,1.4));
  const total=lines.length*lh + (subLines.length?subLines.length*16+6:0);
  let y=cy(b)-total/2+lh-4;
  fg.push(TL(cx(b),y,lines,{cls,fill:tcolor,lh}));
  if(subLines.length){ fg.push(TL(cx(b),y+(lines.length-1)*lh+18,subLines,{cls:'an',fill:subColor||tcolor,lh:16})); }
}
// ---- data box (white, inset title top-left + chips bottom). title: string or [l1,l2] ----
function dataBox(name,title,tcolor,chipColor,n,{stroke=C.gS}={}){
  const b=B[name];
  fg.push(rrect(b.x,b.y,b.w,b.h,9,'#ffffff',stroke,1.3));
  const lines=Array.isArray(title)?title:[title];
  lines.forEach((ln,i)=>fg.push(T(b.x+12,b.y+22+i*18,ln,{cls:'bl',fill:tcolor})));
  fg.push(chips(b.x+12,b.y+b.h-22,n,chipColor));
}

// === DRUG ===
// inputs
(()=>{ const b=B.IN_SMILES; fg.push(rrect(b.x,b.y,b.w,b.h,9,'#fff',C.gS,1.3));
  fg.push(T(b.x+12,b.y+22,'SMILES',{cls:'bl',fill:C.gT}));
  // tiny stylized molecule
  const mx=b.x+30,my=b.y+48,r=12;
  let hex='';for(let i=0;i<6;i++){const a=Math.PI/6+i*Math.PI/3;hex+=(i?'L':'M')+(mx+r*Math.cos(a)).toFixed(1)+' '+(my+r*Math.sin(a)).toFixed(1)+' ';}
  fg.push(`<path d="${hex}Z" fill="none" stroke="#8a99a8" stroke-width="1.6"/>`);
  fg.push(`<line x1="${mx+r}" y1="${my}" x2="${mx+r+16}" y2="${my-8}" stroke="#8a99a8" stroke-width="1.6"/><circle cx="${mx+r+20}" cy="${my-9}" r="3.2" fill="none" stroke="#8a99a8" stroke-width="1.6"/>`);
  fg.push(T(b.x+86,b.y+52,'C C ( = O ) …',{cls:'mono',fill:C.sub}));
})();
(()=>{ const b=B.IN_GRAPH; fg.push(rrect(b.x,b.y,b.w,b.h,9,'#fff',C.gS,1.3));
  fg.push(TL(cx(b),cy(b)-6,['2D Molecular','Graph'],{cls:'bl',fill:C.gT,lh:19})); })();
(()=>{ const b=B.IN_CONF; fg.push(rrect(b.x,b.y,b.w,b.h,9,'#fff',C.gS,1.3));
  fg.push(TL(cx(b),cy(b)-6,['K Conformers','+ Energy'],{cls:'bl',fill:C.gT,lh:19})); })();
mod('EN_CHEM',C.gMod,C.gS,['ChemBERTa'],C.gT);
mod('EN_GCN', C.gMod,C.gS,['GCN'],C.gT);
mod('EN_EGNN',C.gMod,C.gS,['EGNN'],C.gT);
dataBox('OU_SEM','Molecular Semantics',C.gT,C.gChip,5,{stroke:C.gS});
dataBox('OU_TOPO','Topology Tokens',C.gT,C.gChip,5,{stroke:C.gS});
dataBox('OU_CSUM','Conformer Summaries',C.gT,C.gChip,5,{stroke:C.gS});

// === CONFORMER ===
dataBox('CF_PSC','Protein Seed Context',C.bT,C.bChip,4,{stroke:C.oS});
dataBox('CF_EN','Energy Embedding',C.oT,C.oChip,4,{stroke:C.oS});
mod('CF_MLP',C.oMod,C.oS,['Score','MLP'],C.oT,{lh:21});
dataBox('CF_GEO',['Target-aware','Geometry Tokens'],C.oT,C.oChip,3,{stroke:C.oS});
// alpha label + bars
fg.push(T(392,462,sub('α','k')+'  (softmax)',{cls:'bl',fill:C.oT}));
B.AB.forEach(bar=>{ fg.push(rrect(bar.x,bar.y,bar.w,18,3,C.oChip,'#fff',1)); fg.push(T(bar.x+bar.w+10,bar.y+14,bar.v,{cls:'an',fill:C.ink})); });
fg.push(opCircle(B.CF_OPx,B.CF_OPy,B.CF_OPr,'x',C.oT));
// badge 1
badge(B.PN_CONF.x+418,B.PN_CONF.y+12,'Key Contribution',1,288);

// === FUSION ===
dataBox('FU_1D','1D Semantics  '+sub('d','1d'),C.fT,C.gChip,4,{stroke:C.fS});
dataBox('FU_GEO','2D/3D Geometry  '+sub('d','geo'),C.oT,C.oChip,4,{stroke:C.fS});
(()=>{ const b=B.FU_SIG; fg.push(rrect(b.x,b.y,b.w,b.h,10,C.gMod,C.fS,1.4));
  fg.push(T(cx(b),b.y+24,'Sigmoid Gate (per token)',{cls:'bl',fill:C.fT,anchor:'middle'}));
  const vals=['0.15','0.62','0.81','0.33']; const n=4; const sp=b.w/(n+1);
  vals.forEach((v,i)=>{ const ccx=b.x+sp*(i+1), ccy=b.y+56;
    fg.push(`<circle cx="${ccx}" cy="${ccy}" r="14" fill="#fff" stroke="${C.fT}" stroke-width="1.6"/>`);
    fg.push(T(ccx,ccy+5,'σ',{cls:'an',fill:C.fT,anchor:'middle'}));
    fg.push(T(ccx,ccy+30,v,{cls:'tn',fill:C.sub,anchor:'middle'})); });
})();
fg.push(opCircle(B.FU_MGx,B.FU_MGy,B.FU_MGr,'eq',C.fT));
dataBox('FU_FUSED','Fused Drug  D',C.fT,C.gChip,4,{stroke:C.fS});
dataBox('FU_LIG','Ligand Context  '+sub('d','g'),C.fT,C.gChip,4,{stroke:C.fS});

// === PROTEIN ===
(()=>{ const b=B.PR_SEQ; fg.push(rrect(b.x,b.y,b.w,b.h,9,'#fff',C.bS,1.3));
  fg.push(TL(cx(b),b.y+26,['Protein Sequence','/ Structure'],{cls:'bl',fill:C.bT,lh:18}));
  // stylized ribbon (helix coils)
  const rx=b.x+20,ry=b.y+86,rw=b.w-40;
  let p='M '+rx+' '+(ry+18);
  for(let i=0;i<4;i++){ const x0=rx+i*(rw/4); p+=` C ${x0+rw/16} ${ry-6}, ${x0+rw/8} ${ry-6}, ${x0+3*rw/16} ${ry+18} S ${x0+rw/4} ${ry+42}, ${x0+rw/4} ${ry+18}`; }
  fg.push(`<path d="${p}" fill="none" stroke="${C.bChip}" stroke-width="5" stroke-linecap="round" opacity="0.85"/>`);
  fg.push(T(cx(b),ry+58,'ribbon structure',{cls:'tn',fill:C.sub,anchor:'middle'}));
  fg.push(TL(cx(b),b.y+186,['MAVSEQLKVEEL','LSKNYHLENEVAR','LKKLV …'],{cls:'mono',fill:C.sub,lh:24}));
})();
mod('PR_BERT',C.bMod,C.bS,['ProtBERT'],C.bT,{cls:'ml'});
dataBox('PR_P0',['Protein Seed','Tokens '+sub('P','0')],C.bT,C.bChip,3,{stroke:C.bS});
dataBox('PR_PG',['Protein Seed','Context '+sub('p','g')],C.bT,C.bChip,3,{stroke:C.bS});
mod('PR_FILM',C.bMod,C.bS,['Ligand-','conditioned','FiLM'],C.bT,{lh:20,subLines:['γ (scale)','β (shift)'],subColor:C.bT});
mod('PR_ACMIX',C.bMod,C.bS,['ProteinACmix  × 3'],C.bT,{subLines:['( dynamic conv','+ self-attention )'],subColor:C.sub});
dataBox('PR_TOK',['Protein Tokens','P'],C.bT,C.bChip,3,{stroke:C.bS});

// === MAMBA ===
dataBox('MB_IN',['Protein Tokens','P'],C.bT,C.bChip,3,{stroke:C.pS});
mod('MB_SSM',C.pMod,C.pS,['Bidirectional','Mamba (SSM)'],C.pT,{lh:24,subLines:['→ forward scan','← backward scan'],subColor:C.pT});
fg.push(opCircle(B.MB_Gx,B.MB_Gy,B.MB_Gr,'odot',C.pT));
dataBox('MB_REF',['Refined Protein','P′'],C.pT,C.pChip,3,{stroke:C.pS});
fg.push(T(1340,250,'gated residual:  P ← P + σ(g) · ( Mamba(P) − P )',{cls:'fml',fill:C.sub}));
badge(B.PN_MAMBA.x+560,B.PN_MAMBA.y+12,'Key Contribution',2,238);

// === BIINTENTION ===
mod('BI_D2P',C.iMod,C.iS,['Drug → Protein','Cross-Attn  (Q = D)'],C.iT,{lh:24});
mod('BI_P2D',C.iMod,C.iS,['Protein → Drug','Cross-Attn  (Q = P)'],C.iT,{lh:24});
(()=>{ const b=B.BI_MAP; fg.push(rrect(b.x,b.y,b.w,b.h,8,'#fff',C.iS,1.3));
  fg.push(T(b.x+12,b.y+22,'Cross-Attention Map',{cls:'bl',fill:C.iT}));
  const gx=b.x+18,gy=b.y+34,cols=10,rows=5,cw=(b.w-36)/cols,ch=(b.h-46)/rows;
  const pat=[3,1,5,2,8,4,1,6,2,3, 1,7,2,9,3,1,5,2,7,1, 6,2,4,1,3,8,2,5,1,4, 2,9,1,6,2,3,7,1,4,2, 8,1,5,2,7,1,3,6,2,9];
  for(let r=0;r<rows;r++)for(let cc=0;cc<cols;cc++){ const v=pat[r*cols+cc]/9; fg.push(`<rect x="${(gx+cc*cw).toFixed(1)}" y="${(gy+r*ch).toFixed(1)}" width="${(cw-1.5).toFixed(1)}" height="${(ch-1.5).toFixed(1)}" fill="${C.bChip}" opacity="${(0.12+0.85*v).toFixed(2)}"/>`); }
})();
mod('BI_POOL',C.iMod,C.iS,['Max-Pool','over tokens'],C.iT,{lh:24});
dataBox('BI_JOINT','Joint Representation  f   [bs, 256]',C.iT,C.bChip,8,{stroke:C.iS});

// === PREDICTION ===
mod('PD_MLP',C.rMod,C.rS,['MLP Decoder','(FC × 3 + ReLU)'],C.rT,{lh:24});
(()=>{ const b=B.PD_SIG; fg.push(rrect(b.x,b.y,b.w,b.h,12,C.rMod,C.rS,1.4));
  const ax=b.x+40,ay=cy(b)+34,aw=b.w-80,ah=72;
  fg.push(`<line x1="${ax}" y1="${ay}" x2="${ax+aw}" y2="${ay}" stroke="${C.sub}" stroke-width="1.4"/><line x1="${ax}" y1="${ay}" x2="${ax}" y2="${ay-ah}" stroke="${C.sub}" stroke-width="1.4"/>`);
  let s='M '+ax+' '+ay; for(let i=0;i<=20;i++){const t=i/20;const xx=ax+t*aw;const yy=ay-ah/(1+Math.exp(-(t*12-6)));s+=` L ${xx.toFixed(1)} ${yy.toFixed(1)}`;}
  fg.push(`<path d="${s}" fill="none" stroke="${C.rT}" stroke-width="3"/>`);
  fg.push(T(cx(b),b.y+30,'Sigmoid',{cls:'bl',fill:C.rT,anchor:'middle'}));
})();
(()=>{ const b=B.PD_OUT; fg.push(rrect(b.x,b.y,b.w,b.h,12,C.rMod,C.rS,1.4));
  fg.push(T(cx(b),cy(b)-8,'ŷ ∈ [0, 1]',{cls:'ml',fill:C.rT,anchor:'middle'}));
  fg.push(TL(cx(b),cy(b)+22,['Interaction','Probability'],{cls:'bl',fill:C.rT,lh:20})); })();

// ---- badge ----
function badge(x,y,text,num,w){
  const h=30;
  fg.push(rrect(x,y,w,h,15,C.badge,'#e9b91f',1.2));
  fg.push(T(x+16,y+20,'★',{cls:'bl',fill:C.star}));
  fg.push(T(x+38,y+20,esc(text)+'  '+(num===1?'①':'②'),{cls:'bl',fill:C.badgeT}));
}

// ===================================================================
// CONNECTORS
// ===================================================================
const E=(n,s)=>{const b=B[n];return s==='r'?[b.x+b.w,cy(b)]:s==='l'?[b.x,cy(b)]:s==='t'?[cx(b),b.y]:[cx(b),b.y+b.h];};

// drug internal
['SMILES:CHEM','GRAPH:GCN','CONF:EGNN'].forEach(p=>{const [a,b]=p.split(':');mid.push(A([E('IN_'+a,'r'),E('EN_'+b,'l')]));});
['CHEM:SEM','GCN:TOPO','EGNN:CSUM'].forEach(p=>{const [a,b]=p.split(':');mid.push(A([E('EN_'+a,'r'),E('OU_'+b,'l')]));});

// drug -> fusion
mid.push(A([E('OU_SEM','r'),[775,B.OU_SEM.y+35],[775,cy(B.FU_1D)],E('FU_1D','l')]));
mid.push(A([E('OU_TOPO','r'),[768,B.OU_TOPO.y+35],[768,cy(B.FU_GEO)-8],E('FU_GEO','l')]));

// conformer internal
mid.push(A([E('CF_PSC','r'),[230,cy(B.CF_PSC)],[230,cy(B.CF_MLP)],E('CF_MLP','l')]));
mid.push(A([E('CF_EN','r'),[230,cy(B.CF_EN)],[230,cy(B.CF_MLP)],E('CF_MLP','l')]));
mid.push(A([E('CF_MLP','r'),[B.AB[1].x-8,B.AB[1].y+9]]));  // MLP -> alpha bars
mid.push(A([[B.AB[1].x+B.AB[1].w+44,B.AB[1].y+9],[B.CF_OPx-B.CF_OPr,B.CF_OPy]]));
mid.push(A([[B.CF_OPx+B.CF_OPr,B.CF_OPy],E('CF_GEO','l')]));
// conformer summaries -> OPx (down)
mid.push(A([E('OU_CSUM','b'),[cx(B.OU_CSUM),420],[B.CF_OPx,420],[B.CF_OPx,B.CF_OPy-B.CF_OPr]]));
// target-aware geometry -> fusion geo (navy)
mid.push(A([E('CF_GEO','t'),[cx(B.CF_GEO),440],[782,440],[782,cy(B.FU_GEO)+8],E('FU_GEO','l')]));

// modulation (red dashed): protein seed context p_g -> conformer PSC
mid.push(A([E('PR_PG','l'),[360,cy(B.PR_PG)],[24,cy(B.PR_PG)],[24,cy(B.CF_PSC)],E('CF_PSC','l')],{color:C.red,dash:true}));
// modulation: target-aware geometry -> FiLM
mid.push(A([E('CF_GEO','b'),[cx(B.CF_GEO),628],[cx(B.PR_FILM),628],E('PR_FILM','t')],{color:C.red,dash:true}));

// fusion internal
mid.push(A([E('FU_1D','b'),E('FU_GEO','t')]));
mid.push(A([E('FU_GEO','b'),E('FU_SIG','t')]));
mid.push(A([E('FU_SIG','b'),[B.FU_MGx,B.FU_MGy-B.FU_MGr]]));
mid.push(A([[B.FU_MGx,B.FU_MGy+B.FU_MGr],E('FU_FUSED','t')]));
mid.push(A([E('FU_FUSED','b'),E('FU_LIG','t')]));

// fused drug D -> biintention D2P
mid.push(A([E('FU_FUSED','r'),[1108,cy(B.FU_FUSED)],[1108,cy(B.BI_D2P)],E('BI_D2P','l')]));
// ligand context d_g -> FiLM (down-left)
mid.push(A([E('FU_LIG','l'),[770,cy(B.FU_LIG)],[770,790],[cx(B.PR_FILM)+30,790],[cx(B.PR_FILM)+30,B.PR_FILM.y]],{color:C.navy}));

// protein internal
mid.push(A([E('PR_SEQ','r'),E('PR_BERT','l')]));
mid.push(A([E('PR_BERT','r'),[360,cy(B.PR_BERT)],[360,cy(B.PR_P0)],E('PR_P0','l')]));
mid.push(A([E('PR_BERT','r'),[360,cy(B.PR_BERT)],[360,cy(B.PR_PG)],E('PR_PG','l')]));
mid.push(A([E('PR_P0','r'),[548,cy(B.PR_P0)],[548,cy(B.PR_FILM)],E('PR_FILM','l')]));
mid.push(A([E('PR_FILM','r'),E('PR_ACMIX','l')]));
mid.push(A([E('PR_ACMIX','r'),E('PR_TOK','l')]));

// protein tokens P -> mamba in (up-right via channel)
mid.push(A([E('PR_TOK','r'),[1092,cy(B.PR_TOK)],[1092,cy(B.MB_IN)],E('MB_IN','l')]));
// mamba chain
mid.push(A([E('MB_IN','r'),E('MB_SSM','l')]));
mid.push(A([E('MB_SSM','r'),[B.MB_Gx-B.MB_Gr,B.MB_Gy]]));
mid.push(A([[B.MB_Gx+B.MB_Gr,B.MB_Gy],E('MB_REF','l')],{color:C.purple}));
// refined protein P' -> biintention P2D (down)
mid.push(A([E('MB_REF','b'),[cx(B.MB_REF),250],[cx(B.BI_P2D),250],E('BI_P2D','t')],{color:C.purple}));

// biintention internal
mid.push(A([E('BI_D2P','b'),E('BI_MAP','t')]));
mid.push(A([E('BI_P2D','b'),E('BI_POOL','t')]));
mid.push(A([E('BI_MAP','b'),[cx(B.BI_MAP),652],[cx(B.BI_JOINT)-120,652],[cx(B.BI_JOINT)-120,B.BI_JOINT.y]]));
mid.push(A([E('BI_POOL','b'),[cx(B.BI_POOL),648],[cx(B.BI_JOINT)+150,648],[cx(B.BI_JOINT)+150,B.BI_JOINT.y]]));
// joint -> prediction
mid.push(A([E('BI_JOINT','b'),[cx(B.BI_JOINT),736],[cx(B.PD_MLP),736],E('PD_MLP','t')]));
// prediction chain
mid.push(A([E('PD_MLP','r'),E('PD_SIG','l')]));
mid.push(A([E('PD_SIG','r'),E('PD_OUT','l')]));

// ---- LEGEND (horizontal footer) ----
(()=>{ const y=1074;
  fg.push(T(300,y,'Legend',{cls:'bl',fill:C.ink}));
  fg.push(A([[394,y-5],[440,y-5]]));                 fg.push(T(450,y,'data flow',{cls:'an',fill:C.ink}));
  fg.push(A([[616,y-5],[662,y-5]],{color:C.red,dash:true})); fg.push(T(672,y,'target-aware modulation',{cls:'an',fill:C.red}));
  fg.push(opCircle(968,y-6,12,'odot',C.pT));         fg.push(T(990,y,'gated fusion',{cls:'an',fill:C.ink}));
  fg.push(T(1170,y,'★',{cls:'bl',fill:C.star}));      fg.push(T(1192,y,'our key contribution',{cls:'an',fill:C.oT}));
})();

// ===================================================================
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="Arial, Helvetica, sans-serif">
<defs>
  ${['aN:'+C.navy,'aR:'+C.red,'aP:'+C.purple].map(m=>{const[id,col]=m.split(':');return `<marker id="${id}" markerUnits="userSpaceOnUse" markerWidth="13" markerHeight="13" refX="9.5" refY="5.2" orient="auto"><path d="M0,0.6 L11,5.2 L0,9.8 Z" fill="${col}"/></marker>`;}).join('')}
</defs>
<style>
  text{font-family:Arial,Helvetica,sans-serif;}
  .pt{font-size:22px;font-weight:700;}
  .ml{font-size:18px;font-weight:700;}
  .bl{font-size:15px;font-weight:700;}
  .an{font-size:13px;font-weight:400;}
  .tn{font-size:12px;font-weight:400;}
  .fml{font-size:14px;font-weight:400;}
  .mono{font-size:13px;font-weight:400;font-family:'Consolas','Courier New',monospace;}
  .subsc{font-size:11px;font-weight:700;}
</style>
<rect width="${W}" height="${H}" fill="#ffffff"/>
${bg.join('\n')}
${mid.join('\n')}
${fg.join('\n')}
</svg>`;

fs.writeFileSync(__dirname+'/tamr_dti_architecture_v8.svg', svg);
console.log('wrote tamr_dti_architecture_v8.svg', svg.length, 'bytes');
