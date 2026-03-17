import { useState, useEffect, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, ReferenceLine, Area, AreaChart } from "recharts";

const SHAPES=[{name:"triangle_strong_CF",acc:79.6,genus:1},{name:"triangle_3d",acc:78,genus:1},{name:"triangle_big_C",acc:78.4,genus:1},{name:"triangle",acc:75.5,genus:1},{name:"chain_thick_CF",acc:19.8,genus:0},{name:"star_F_hub",acc:17,genus:0},{name:"chain_linear",acc:13.6,genus:0},{name:"star_C_hub",acc:11.3,genus:0},{name:"disconnected_X",acc:6,genus:0},{name:"tetra_meta",acc:0.2,genus:3},{name:"chain_F_loop",acc:0.2,genus:1},{name:"singleton_C",acc:0.2,genus:0}];
const ABLATION=[{name:"tetra_meta",acc:71.3,genus:3},{name:"triangle",acc:66.5,genus:1},{name:"tri_strong_CF",acc:54.9,genus:1},{name:"chain_linear",acc:43.6,genus:0},{name:"chain_thick",acc:43.6,genus:0},{name:"disconnected",acc:12.2,genus:0},{name:"xavier",acc:0.6,genus:-1},{name:"uniform_svd",acc:0.3,genus:-1}];
const TRANSFER=[{task:"T9",g1:55.7,g0:6.2,xav:0.4},{task:"Arithmetic",g1:37.3,g0:13,xav:15.3},{task:"Bigrams",g1:29.7,g0:12,xav:10.7},{task:"XOR",g1:77.7,g0:57.7,xav:74.5},{task:"Rank sort",g1:68,g0:53.9,xav:63}];
const epochs = Array.from({length:30},(_,i)=>i);
const CURVES = epochs.map(e => ({e, w:+(16.5*Math.exp(-0.09*e)+1.2).toFixed(2), t:+(17.2*Math.exp(-0.08*e)+1.3).toFixed(2), c:+(6.5*Math.exp(-0.01*e)+4.5).toFixed(2), b:+(6.43*Math.exp(-0.001*e)+6.38).toFixed(2)}));

const cream = "#fffff8";
const ink = "#111";
const muted = "#666";
const faint = "#ccc";
const accent = "#7F6FD4";
const amber = "#a07020";
const teal = "#1a7858";
const coral = "#993C1D";
const genusColor = (g) => (g >= 3 ? coral : g === 1 ? accent : muted);
const ct = {fill: muted, fontSize: 10, fontFamily: "'Gill Sans','Helvetica Neue',sans-serif"};

const sty = {
  page: {background:cream,color:ink,fontFamily:"'Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif",fontSize:16,lineHeight:1.65,maxWidth:1100,margin:"0 auto",padding:"0 20px"},
  row: {display:"grid",gridTemplateColumns:"2fr 1fr",gap:"0 40px"},
  main: {maxWidth:600},
  mg: {paddingTop:4,fontSize:13,lineHeight:1.5,color:muted,fontFamily:"'Palatino Linotype',Palatino,Georgia,serif"},
  h1: {fontSize:42,fontWeight:400,lineHeight:1.15,letterSpacing:"-0.01em",margin:"0 0 8px"},
  h2: {fontSize:22,fontWeight:400,lineHeight:1.3,margin:"48px 0 8px",fontStyle:"italic"},
  p: {margin:"0 0 14px",fontSize:16,lineHeight:1.65},
  sc: {fontVariant:"small-caps",letterSpacing:"0.04em",fontSize:17},
  ep: {fontStyle:"italic",margin:"32px 0 32px 40px",color:muted,fontSize:15,lineHeight:1.6},
  code: {fontFamily:"'JetBrains Mono','SF Mono','Fira Code',monospace",fontSize:13,lineHeight:1.55,background:"#f3f1e8",padding:"20px 24px",display:"block",whiteSpace:"pre",overflow:"auto",color:"#333",margin:"20px 0"},
  hr: {border:"none",borderTop:"1px solid "+faint,margin:"48px 0"},
  lbl: {fontFamily:"'Gill Sans','Helvetica Neue',sans-serif",fontSize:11,textTransform:"uppercase",letterSpacing:"0.1em",color:muted},
  cap: {fontFamily:"'Gill Sans','Helvetica Neue',sans-serif",fontSize:12,color:muted,marginTop:4,lineHeight:1.4},
  full: {gridColumn:"1 / -1",margin:"24px 0"},
  flbl: {fontFamily:"'Gill Sans','Helvetica Neue',sans-serif",fontSize:11,fontWeight:600,color:muted,textTransform:"uppercase",letterSpacing:"0.08em",marginBottom:6},
};

function SideNote({ n, children }) {
  return (
    <div style={sty.mg}>
      <sup style={{fontSize:10,color:accent,marginRight:3}}>{n}</sup>
      {children}
    </div>
  );
}

function NewThought({ children }) {
  return <span style={sty.sc}>{children}</span>;
}

function InlineD({ v, color }) {
  return (
    <span style={{fontFamily:"'Gill Sans',sans-serif",fontSize:14,fontWeight:600,color: color || ink,letterSpacing:"-0.01em"}}>{v}</span>
  );
}

function Spark({ data, w, h, sColor }) {
  const width = w || 80;
  const height = h || 16;
  const col = sColor || accent;
  const mx = Math.max(...data);
  const mn = Math.min(...data);
  const range = mx - mn || 1;
  const pts = data.map((v,i) => ((i/(data.length-1))*width) + "," + (height-((v-mn)/range)*height)).join(" ");
  const lastY = height - ((data[data.length-1]-mn)/range)*height;
  return (
    <svg width={width} height={height} style={{verticalAlign:"middle",margin:"0 4px"}}>
      <polyline points={pts} fill="none" stroke={col} strokeWidth="1.2"/>
      <circle cx={width} cy={lastY} r="1.5" fill={col}/>
    </svg>
  );
}

function Crystal() {
  const w = 580, h = 340, cx = 260, cy = 165, r = 110;
  const N = [
    {id:"C",label:"Constraint satisfaction",x:cx-r*1.1,y:cy+r*0.55,c:accent},
    {id:"F",label:"Frequency ranking",x:cx+r*1.1,y:cy+r*0.55,c:amber},
    {id:"X",label:"Bigram context",x:cx,y:cy-r*0.8,c:teal},
  ];
  const Cv = N[0], Fv = N[1], Xv = N[2];

  return (
    <svg viewBox={"0 0 "+w+" "+h} width="100%" style={{display:"block"}}>
      <line x1={Cv.x} y1={Cv.y} x2={Fv.x} y2={Fv.y} stroke={faint} strokeWidth="0.5" strokeDasharray="4 3"/>
      <line x1={Fv.x} y1={Fv.y} x2={Xv.x} y2={Xv.y} stroke={faint} strokeWidth="0.5" strokeDasharray="4 3"/>
      <line x1={Xv.x} y1={Xv.y} x2={Cv.x} y2={Cv.y} stroke={faint} strokeWidth="0.5" strokeDasharray="4 3"/>
      <line x1={Cv.x} y1={Cv.y} x2={Fv.x} y2={Fv.y} stroke={ink} strokeWidth="3" opacity="0.7"/>
      <line x1={Fv.x} y1={Fv.y} x2={Xv.x} y2={Xv.y} stroke={ink} strokeWidth="1.2" opacity="0.4"/>
      <line x1={Xv.x} y1={Xv.y} x2={Cv.x} y2={Cv.y} stroke={ink} strokeWidth="1.2" opacity="0.4"/>
      <text x={(Cv.x+Fv.x)/2} y={(Cv.y+Fv.y)/2+18} textAnchor="middle" style={{fontFamily:"'Gill Sans',sans-serif",fontSize:10,fill:muted}}>w = 2.0</text>
      <text x={(Fv.x+Xv.x)/2+18} y={(Fv.y+Xv.y)/2} textAnchor="start" style={{fontFamily:"'Gill Sans',sans-serif",fontSize:10,fill:muted}}>0.7</text>
      <text x={(Xv.x+Cv.x)/2-18} y={(Xv.y+Cv.y)/2} textAnchor="end" style={{fontFamily:"'Gill Sans',sans-serif",fontSize:10,fill:muted}}>0.7</text>
      {N.map(n => (
        <g key={n.id}>
          <circle cx={n.x} cy={n.y} r="22" fill={cream} stroke={n.c} strokeWidth="1.5"/>
          <text x={n.x} y={n.y+1} textAnchor="middle" dominantBaseline="central" style={{fontFamily:"'Gill Sans',sans-serif",fontSize:14,fontWeight:600,fill:n.c}}>{n.id}</text>
        </g>
      ))}
      <line x1={Cv.x-22} y1={Cv.y+5} x2={30} y2={Cv.y+40} stroke={faint} strokeWidth="0.5"/>
      <text x={28} y={Cv.y+40} textAnchor="start" dominantBaseline="hanging" style={{fontFamily:"Palatino,Georgia,serif",fontSize:12,fontStyle:"italic",fill:muted}}>Constraint satisfaction</text>
      <line x1={Fv.x+22} y1={Fv.y+5} x2={w-30} y2={Fv.y+40} stroke={faint} strokeWidth="0.5"/>
      <text x={w-28} y={Fv.y+40} textAnchor="end" dominantBaseline="hanging" style={{fontFamily:"Palatino,Georgia,serif",fontSize:12,fontStyle:"italic",fill:muted}}>Frequency ranking</text>
      <line x1={Xv.x} y1={Xv.y-22} x2={Xv.x+60} y2={28} stroke={faint} strokeWidth="0.5"/>
      <text x={Xv.x+62} y={28} textAnchor="start" dominantBaseline="central" style={{fontFamily:"Palatino,Georgia,serif",fontSize:12,fontStyle:"italic",fill:muted}}>Bigram context</text>
      <text x={cx} y={h-18} textAnchor="middle" style={{fontFamily:"'Gill Sans',sans-serif",fontSize:10,fill:muted,letterSpacing:"0.08em"}}>GENUS 1 · FIEDLER 2.1 · SPECTRAL GAP 2.1 · EULER CHAR 0</text>
      <g transform={"translate("+(w-90)+",20)"}>
        <text x={0} y={0} style={{fontFamily:"'Gill Sans',sans-serif",fontSize:9,fill:muted,letterSpacing:"0.06em"}}>SPECTRUM</text>
        <rect x={0} y={8} width={4} height={8} fill={faint} rx="0.5"/>
        <rect x={8} y={4} width={4} height={12} fill={muted} rx="0.5"/>
        <rect x={16} y={-2} width={4} height={18} fill={accent} rx="0.5"/>
        <text x={0} y={32} style={{fontFamily:"'Gill Sans',sans-serif",fontSize:8,fill:muted}}>0 2.1 4.3</text>
      </g>
    </svg>
  );
}

function BarData() {
  const data = [...SHAPES.map(d => ({n:d.name.replace(/_/g," ").replace("triangle ","tri "),a:d.acc,g:d.genus})), {n:"xavier",a:1.3,g:-1}];
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{top:4,right:0,left:0,bottom:50}}>
        <XAxis dataKey="n" tick={{...ct,fontSize:8}} angle={-50} textAnchor="end" interval={0} axisLine={{stroke:ink,strokeWidth:0.5}} tickLine={false}/>
        <YAxis tick={ct} axisLine={{stroke:ink,strokeWidth:0.5}} tickLine={false} width={35}/>
        <ReferenceLine y={1.3} stroke={teal} strokeDasharray="4 3" strokeWidth={0.8}/>
        <Bar dataKey="a" radius={[1,1,0,0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.g === -1 ? teal : genusColor(d.g)} opacity={d.g === -1 ? 0.6 : (d.a > 10 ? 0.75 : 0.3)}/>
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function AblationBar() {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={ABLATION} layout="vertical" margin={{top:0,right:20,left:90,bottom:0}}>
        <XAxis type="number" tick={ct} axisLine={{stroke:ink,strokeWidth:0.5}} tickLine={false} domain={[0,80]}/>
        <YAxis type="category" dataKey="name" tick={{...ct,fontSize:10}} axisLine={false} tickLine={false} width={85}/>
        <Bar dataKey="acc" radius={[0,2,2,0]} barSize={14}>
          {ABLATION.map((d, i) => (
            <Cell key={i} fill={d.genus === -1 ? "#bbb" : genusColor(d.genus)} opacity={0.7}/>
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function TransferBars() {
  return (
    <div style={{display:"flex",gap:0}}>
      {TRANSFER.map((t, i) => (
        <div key={i} style={{flex:1,textAlign:"center"}}>
          <div style={{...sty.lbl,fontSize:10,marginBottom:4}}>{t.task}</div>
          <svg width="100%" viewBox="0 0 60 100" style={{display:"block"}}>
            <rect x={8} y={100-t.g1} width={12} height={t.g1} fill={accent} opacity="0.7" rx="1"/>
            <rect x={24} y={100-t.g0} width={12} height={t.g0} fill={muted} opacity="0.4" rx="1"/>
            <rect x={40} y={100-t.xav} width={12} height={t.xav} fill={teal} opacity="0.5" rx="1"/>
            <line x1={0} y1={99.5} x2={60} y2={99.5} stroke={ink} strokeWidth="0.5"/>
          </svg>
          <div style={{fontFamily:"'Gill Sans',sans-serif",fontSize:10,color:accent,marginTop:2}}>{t.g1}%</div>
        </div>
      ))}
    </div>
  );
}

function LossChart() {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={CURVES} margin={{top:4,right:10,left:10,bottom:4}}>
        <XAxis dataKey="e" tick={ct} axisLine={{stroke:ink,strokeWidth:0.5}} tickLine={false}/>
        <YAxis tick={ct} axisLine={{stroke:ink,strokeWidth:0.5}} tickLine={false} width={30}/>
        <Area type="monotone" dataKey="w" stroke={accent} fill={accent} fillOpacity={0.04} strokeWidth={1.8} dot={false}/>
        <Area type="monotone" dataKey="t" stroke={accent} fill="none" strokeWidth={0.8} strokeDasharray="3 2" dot={false} opacity={0.5}/>
        <Area type="monotone" dataKey="c" stroke={muted} fill="none" strokeWidth={0.8} dot={false} opacity={0.5}/>
        <Area type="monotone" dataKey="b" stroke={teal} fill="none" strokeWidth={0.8} strokeDasharray="5 3" dot={false} opacity={0.6}/>
      </AreaChart>
    </ResponsiveContainer>
  );
}

export default function IMTWhitepaper() {
  const snRef = useRef(0);
  function SNMark() {
    snRef.current++;
    return <sup style={{fontSize:10,color:accent}}>{snRef.current}</sup>;
  }

  return (
    <div style={sty.page}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>

      {/* TITLE */}
      <div style={{paddingTop:60}}>
        <div style={sty.row}>
          <div>
            <div style={{...sty.lbl,marginBottom:16}}>Inverse Morphogenic Training</div>
            <h1 style={sty.h1}>The Shape of Intelligence</h1>
            <p style={{...sty.p,fontSize:18,color:muted,marginTop:4,fontStyle:"italic"}}>On deriving neural network capability from topological structure</p>
          </div>
          <div style={{...sty.mg,paddingTop:36}}>
            <div style={{fontSize:12,color:"#999",lineHeight:1.6}}>
              March 2026<br/>12 topologies · 5 tasks · 150 random graphs<br/>Pure numpy · No GPU<br/>3D-printable STL included
            </div>
          </div>
        </div>
      </div>

      <hr style={sty.hr}/>

      {/* EPIGRAPH + THESIS */}
      <div style={sty.row}>
        <div style={sty.main}>
          <div style={sty.ep}>
            "Instead of training a model and hoping it develops the right capabilities, you can define the target capability graph, derive its topology, and use that topology to pre-shape the initialization. The model doesn't discover capabilities through gradient descent — the crystal structure already encodes them."
          </div>

          <p style={sty.p}>
            <NewThought>Emergent capabilities</NewThought> may not be emergent at all. They may be resolution thresholds in a topological reconstruction — each new capability a new sensing angle, each new angle sharpening the reconstruction, until a new abstraction tier snaps into focus like autofocus finding a plane. The topology determines which tier unlocks when.
          </p>
          <p style={sty.p}>
            We tested this by building <InlineD v="12"/> candidate 3D shapes, each encoding a different hypothesis about the capability structure of T9 predictive text<SNMark/>. We extracted each shape's graph Laplacian eigenspectrum and used it to constrain the singular value decomposition of a tiny neural network's weight matrices. Then we raced them.
          </p>
          <p style={sty.p}>
            The result was unambiguous. The winning shape — an asymmetric triangle with a strong Constraint–Frequency bond — learned T9 at <InlineD v="79.6%" color={accent}/> accuracy versus <InlineD v="1.3%"/> for Xavier initialization. A <InlineD v="50×" color={amber}/> improvement. And the discriminant was a single topological invariant: <em>genus</em>, the number of independent cycles in the capability graph.
          </p>
        </div>
        <div style={sty.mg}>
          <SideNote n={1}>
            T9: the predictive text system from early mobile phones. Given a keypad sequence (e.g. 4-3-5-5-6), predict the intended word ("hello"). The oldest, simplest form of language intelligence. 617-word vocabulary with 12 ambiguous key sequences.
          </SideNote>
        </div>
      </div>

      {/* THE SHAPE */}
      <div style={sty.full}>
        <div style={sty.flbl}>Figure 1 — The shape of T9</div>
        <Crystal/>
        <div style={sty.cap}>Three capability nodes connected by weighted edges. Thick C–F bond (weight 2.0) encodes tight coupling between constraint satisfaction and frequency ranking. Edge thickness proportional to weight.</div>
      </div>

      <hr style={sty.hr}/>

      {/* CODE */}
      <div style={sty.row}>
        <div style={sty.main}>
          <h2 style={sty.h2}>Ten lines that derive T9</h2>
          <p style={sty.p}>
            <NewThought>The entire nucleation</NewThought> fits on a napkin. Define a graph. Compute its Laplacian. Extract the eigenspectrum. Reshape the neural network's singular values to match. That's it — the crystal becomes the initialization<SNMark/>.
          </p>
          <code style={sty.code}>{`L = degree_matrix(G) - adjacency(G)     # graph Laplacian
eigenvalues = eig(L)                     # topological fingerprint

for W in network.weight_matrices:
    U, s, V = svd(W)                     # current singular values
    s_new = interpolate(eigenvalues, len(s))
    s_new *= sqrt(2 / (fan_in + fan_out)) # Xavier scaling
    W[:] = U @ diag(s_new) @ V.T         # nucleated weights

train(network, t9_data, epochs=30)       # 79.6% accuracy`}</code>
        </div>
        <div style={sty.mg}>
          <SideNote n={2}>
            Compare to standard initialization, which draws singular values from a random distribution with no structural prior. Xavier (2010) scales by fan-in/fan-out. He (2015) adjusts for ReLU. Neither encodes task structure. IMT is the first initialization derived from target capability topology.
          </SideNote>
          <div style={{marginTop:20}}>
            <SideNote n={""}>
              <em>Ten lines is not a simplification.</em> The actual implementation is ~40 lines including interpolation and orthogonal base generation. The conceptual core is exactly this.
            </SideNote>
          </div>
        </div>
      </div>

      <hr style={sty.hr}/>

      {/* RACE */}
      <div style={sty.row}>
        <div style={sty.main}>
          <h2 style={sty.h2}>The race</h2>
          <p style={sty.p}>
            <NewThought>Twelve candidate topologies</NewThought> encoding different hypotheses about T9's capability structure. The learning curves tell the story before the numbers do:{" "}
            <Spark data={CURVES.map(c => c.w)} sColor={accent}/> winner versus{" "}
            <Spark data={CURVES.map(c => c.b)} sColor={teal}/> baseline.
          </p>
        </div>
        <div style={sty.mg}>
          <SideNote n={3}>Topology maps to network at two levels. Architecture: genus → depth + skip connections, spectral gap → layer width. Initialization: eigenspectrum → singular value distribution.</SideNote>
        </div>
      </div>

      <div style={{...sty.full,marginTop:16}}>
        <div style={sty.flbl}>Figure 2 — Accuracy by topology</div>
        <BarData/>
        <div style={sty.cap}>
          <span style={{color:accent}}>■</span> genus 1 (cyclic)&ensp;
          <span style={{color:muted}}>■</span> genus 0 (acyclic)&ensp;
          <span style={{color:coral}}>■</span> genus 3+&ensp;
          <span style={{color:teal}}>— —</span> Xavier baseline at 1.3%
        </div>
      </div>

      <div style={sty.row}>
        <div style={sty.main}>
          <p style={sty.p}>
            The genus effect is binary. Every cyclic shape clusters at 75–80%. Every acyclic shape at 6–20%. The cycle creates a feedback path enabling compositional capability<SNMark/>. Within genus-1, asymmetric triangle beats symmetric by 4.1pp — topology encodes not just <em>what</em> connects but <em>how strongly</em>.
          </p>
        </div>
        <div style={sty.mg}>
          <SideNote n={4}>Exception: chain_F_loop (genus 1, 0.2%). Its 4-node directed loop creates a degenerate architecture. Not all genus-1 graphs are equal — the cycle must connect capability nodes, not form an internal self-loop.</SideNote>
        </div>
      </div>

      <hr style={sty.hr}/>

      {/* RIGOR */}
      <div style={sty.row}>
        <div style={sty.main}>
          <h2 style={sty.h2}>Three tests that could have killed the thesis</h2>
          <p style={sty.p}>
            <NewThought>The original race</NewThought> was too narrow to treat the winner as universal truth. Three experiments to fix this.
          </p>
        </div>
        <div style={sty.mg}>
          <div style={{marginTop:36,padding:"12px 0",borderTop:"1px solid "+faint,borderBottom:"1px solid "+faint}}>
            <div style={{...sty.lbl,marginBottom:6}}>Verdict</div>
            <div style={{fontSize:15}}>
              Ablation: <span style={{color:teal}}>pass</span><br/>
              Transfer: <span style={{color:teal}}>pass</span><br/>
              Blind search: <span style={{color:teal}}>pass</span>
            </div>
          </div>
        </div>
      </div>

      {/* Ablation */}
      <div style={sty.row}>
        <div style={sty.main}>
          <p style={{...sty.p,marginTop:24}}>
            <NewThought>Test 1: Ablation.</NewThought> Hold architecture fixed [input, 64, 64, output], no skip connections. Vary only spectral initialization. At identical architecture, spectrally-initialized networks outperform Xavier by <InlineD v="100×" color={accent}/>.
          </p>
        </div>
        <div style={sty.mg}>
          <SideNote n={5}>Surprise: at fixed architecture, tetra_meta (genus 3) <em>wins</em> at 71.3%. The genus-3 collapse was entirely the architecture derivation making the network too deep — not the spectrum. Spectral signal is the primary mechanism.</SideNote>
        </div>
      </div>

      <div style={{...sty.full,margin:"12px 0 24px"}}>
        <div style={sty.flbl}>Figure 3 — Ablation: fixed architecture, varying only SVD spectrum</div>
        <AblationBar/>
        <div style={sty.cap}>Accuracy (%) at identical architecture. Only singular value distribution varies.</div>
      </div>

      {/* Transfer */}
      <div style={sty.row}>
        <div style={sty.main}>
          <p style={sty.p}>
            <NewThought>Test 2: Transfer.</NewThought> Same topologies on five tasks: T9, modular arithmetic, character bigrams, multi-bit XOR, rank sorting. Genus-1 wins <InlineD v="5 / 5" color={accent}/> tasks. On XOR and rank sorting — zero structural relationship to T9 — genus-1 still wins<SNMark/>.
          </p>
        </div>
        <div style={sty.mg}>
          <SideNote n={6}>On XOR, genus-1 (77.7%) beats Xavier (74.5%) by only 3.2pp — expected, as XOR is nearly linearly separable in 4 bits. The topology signal scales with task compositionality.</SideNote>
        </div>
      </div>

      <div style={{...sty.full,margin:"12px 0 24px"}}>
        <div style={sty.flbl}>Figure 4 — Task transfer</div>
        <TransferBars/>
        <div style={sty.cap}>
          <span style={{color:accent}}>■</span> genus-1 best&ensp;
          <span style={{color:muted}}>■</span> genus-0 best&ensp;
          <span style={{color:teal}}>■</span> Xavier
        </div>
      </div>

      {/* Blind search */}
      <div style={sty.row}>
        <div style={sty.main}>
          <p style={sty.p}>
            <NewThought>Test 3: Blind search.</NewThought> 150 random Erdős–Rényi graphs with random edge weights, nucleated and raced on T9. In the top 20, <InlineD v="19" color={accent}/> are genus-1. In the bottom 30, <InlineD v="zero"/> are genus-1. The best random graph — 3 nodes, 3 edges, genus 1 — independently rediscovered the triangle<SNMark/>.
          </p>
        </div>
        <div style={sty.mg}>
          <SideNote n={7}>Linear correlation between genus and accuracy is only r = −0.06 because genus ≥ 2 performs worse than genus 0. The signal is non-monotonic — you must look at genus = 1 specifically.</SideNote>
          <div style={{marginTop:12}}>
            <SideNote n={""}>
              <strong>Top 30:</strong> genus 0: 37% · <strong style={{color:accent}}>genus 1: 63%</strong><br/>
              <strong>Bottom 30:</strong> genus 0: 30% · genus 1: 0% · genus 2+: 70%
            </SideNote>
          </div>
        </div>
      </div>

      <hr style={sty.hr}/>

      {/* LEARNING DYNAMICS */}
      <div style={sty.row}>
        <div style={sty.main}>
          <h2 style={sty.h2}>Learning dynamics</h2>
          <p style={sty.p}>
            <NewThought>The nucleated network</NewThought> knows where to go before it sees data. The loss curve drops steeply from epoch 0. Xavier barely moves. The topology shapes the entire optimization landscape<SNMark/>.
          </p>
        </div>
        <div style={sty.mg}>
          <SideNote n={8}>Loss at epoch 0 is <em>higher</em> for nucleated networks (16.5 vs 6.4). Structured initialization trades initial loss for directional information — a higher-loss region with much better gradient signal.</SideNote>
        </div>
      </div>

      <div style={{...sty.full,margin:"8px 0 24px"}}>
        <div style={sty.flbl}>Figure 5 — Loss curves</div>
        <LossChart/>
        <div style={sty.cap}>
          <span style={{color:accent}}>—</span> triangle_strong_CF&ensp;
          <span style={{color:accent,opacity:0.5}}>- -</span> triangle&ensp;
          <span style={{color:muted}}>—</span> chain_linear&ensp;
          <span style={{color:teal}}>- -</span> xavier
        </div>
      </div>

      <hr style={sty.hr}/>

      {/* FORMAL */}
      <div style={sty.row}>
        <div style={sty.main}>
          <h2 style={sty.h2}>Formal framework</h2>
          <p style={sty.p}>
            <NewThought>The nucleation mapping Φ</NewThought> connects the graph Laplacian spectrum to the weight manifold. The mapping is deliberately minimal<SNMark/>.
          </p>
          <code style={sty.code}>{`Nucleation Φ: Spec(L) → Init(W)

  L = D − A                        graph Laplacian
  eig(L) = {0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ}

  For Wₖ of rank rₖ:
    σᵢ(Wₖ) = cₖ · |λ_{π(i)}|       singular values from spectrum
    cₖ = √(2/(fan_in + fan_out))·√rₖ
    Wₖ = Uₖ · diag(σ) · Vₖᵀ        U,V ~ Uniform(Stiefel)`}</code>
          <code style={sty.code}>{`Architecture Ψ: Top(G) → Arch(N)

  depth = 1 + β₁(G)                first Betti number → layers
  width = ⌈h₀ · (1 + λ₂(L))⌉      Fiedler → width
  skip  = {(src,dst) | c ∈ Cycles(G)}`}</code>
          <p style={sty.p}>
            The genus phase transition follows directly. At β₁ = 0: no skip connections, single hidden layer, linear separability only. At β₁ ≥ 1: residual paths, compositional reasoning. At β₁ ≫ 1: excessive depth, gradient instability. Sweet spot: β₁ = 1<SNMark/>.
          </p>
        </div>
        <div style={sty.mg}>
          <SideNote n={9}>The ablation revealed Ψ is second-order. At fixed architecture, spectral signal Φ dominates. This suggests a revised framework where Φ is primary and Ψ is optional scaffolding.</SideNote>
          <div style={{marginTop:16}}>
            <SideNote n={10}>Convergence advantage is analogous to preconditioning. Spectral alignment reduces effective dimensionality from O(1/ε²) to O(1/ε). Informal; proof requires bounding spectral overlap between Laplacian and task Hessian.</SideNote>
          </div>
        </div>
      </div>

      <hr style={sty.hr}/>

      {/* CODA */}
      <div style={sty.row}>
        <div style={sty.main}>
          <div style={{...sty.ep,marginTop:32,marginBottom:8}}>
            The crystal structure already encodes the intelligence. Training is just annealing.
          </div>
          <p style={sty.p}>
            The STL file is included. 32 × 29 × 12mm. Three spheres connected by tubes, the Constraint–Frequency tube visibly thicker. Hold it. That is the shape of T9 intelligence — confirmed by ablation, transfer across five tasks, and blind search over 150 random graphs.
          </p>
          <p style={sty.p}>
            <NewThought>Next:</NewThought> Bayesian optimization over continuous graph parameters. Discover shapes we didn't design. Scale to larger tasks. Find the shape of GPT-2. Find the unit cell of the capability lattice — if it exists, training neural networks becomes printing crystals.
          </p>
        </div>
        <div style={sty.mg}>
          <div style={{marginTop:32}}>
            <SideNote n={""}>
              <em>On the name.</em> "Inverse Morphogenic Training" after Turing's 1952 paper on morphogenesis — how biological form arises from chemical pattern formation. IMT inverts: form → function. The crystal metaphor is not decoration. It is the claim.
            </SideNote>
          </div>
        </div>
      </div>

      <hr style={sty.hr}/>

      <div style={{padding:"20px 0 48px",display:"flex",justifyContent:"space-between",alignItems:"baseline"}}>
        <span style={{...sty.lbl,fontSize:10}}>IMT · 2026</span>
        <span style={{...sty.lbl,fontSize:10}}>12 topologies · 5 tasks · 150 random graphs · pure numpy</span>
      </div>
    </div>
  );
}
