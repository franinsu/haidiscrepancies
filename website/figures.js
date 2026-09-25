"use strict";

// All marks use frozen figure values. Hover changes emphasis, never the data.
(() => {
  const byId = id => document.getElementById(id);
  const node = (tag, className = "", text = "") => {
    const result = document.createElement(tag);
    result.className = className;
    result.textContent = text;
    return result;
  };
  const svgNode = (tag, attrs = {}, text = "") => {
    const result = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => result.setAttribute(key,
      key === "fill" || key === "stroke" ? window.StudyTheme.color(value) : value));
    result.textContent = text;
    return result;
  };
  const number = x => x !== 0 && Math.abs(x) < .001 ? x.toExponential(1) : x.toFixed(3);
  const estimate = row => `${number(row.mean ?? row.estimate)} [95% CI ${row.ci95.map(number).join(", ")}]`;
  const hasPointer = event => event?.clientX !== undefined && !(event.type === "click" && event.detail === 0);
  const familyColors = ["#CC79A7", "#E6AB02", "#6A3D9A", "#8C564B", "#17AFC2"];
  const pairKey = pair => `${pair.a}-${pair.b}`;

  function hoverScope(root) {
    const marks = [];
    const bindings = new WeakMap();
    let activeKey;
    const tooltip = node("div", "chart-tooltip");
    tooltip.hidden = true;
    tooltip.setAttribute("role", "status");
    root.append(tooltip);
    const scope = {
      onChange: () => {},
      mark(element, tags) { marks.push({element,tags}); element.classList.add("reactive-mark"); },
      emphasize(active) {
        const key=JSON.stringify(active || null);
        if(key===activeKey)return;
        activeKey=key;
        marks.forEach(({element,tags}) => {
          let match = true;
          let related = false;
          // A pair is more specific than its two sources. Unrelated panels stay visible.
          if (active?.puzzle !== undefined && tags.puzzle !== undefined) { related = true; match = active.puzzle === tags.puzzle; }
          else if (active?.pair !== undefined && tags.pair !== undefined) { related = true; match = active.pair === tags.pair; }
          else if (active?.family !== undefined && tags.family !== undefined) { related = true; match = active.family === tags.family; }
          else if (active?.sources && tags.sources) { related = true; match = tags.sources.some(s => active.sources.includes(s)); }
          element.classList.toggle("is-muted", !match);
          element.classList.toggle("is-emphasized", related && match);
        });
        scope.onChange(active);
      },
      clear() { scope.emphasize(null); tooltip.hidden = true; },
      show(element, tags, text, event) {
        scope.emphasize(tags);
        tooltip.textContent = text;
        tooltip.hidden = false;
        const box = element.getBoundingClientRect();
        const px = hasPointer(event) ? event.clientX : box.left + box.width / 2;
        const py = hasPointer(event) ? event.clientY : box.top;
        const width = tooltip.offsetWidth, height = tooltip.offsetHeight;
        tooltip.style.left = Math.max(10, Math.min(window.innerWidth-width-10,px+16)) + "px";
        tooltip.style.top = Math.max(10, py+height+22 < window.innerHeight ? py+18 : py-height-12) + "px";
      },
      bind(element, tags, description) {
        const text = event => typeof description === "function" ? description(event) : description;
        element.setAttribute("aria-label", text());
        const show = event => scope.show(element,tags,text(event),event);
        element.addEventListener("pointerenter", show);
        element.addEventListener("pointermove", show);
        element.addEventListener("pointerleave", () => scope.clear());
        element.addEventListener("focus", () => show());
        element.addEventListener("blur", () => scope.clear());
        element.addEventListener("click", show);
        element.addEventListener("keydown", event => { if (event.key === "Escape") scope.clear(); });
        bindings.set(element,show);
        return show;
      },
    };
    root.addEventListener("pointerleave", () => scope.clear());
    window.addEventListener("scroll", () => scope.clear(), {passive:true});
    window.addEventListener("resize", () => scope.clear(), {passive:true});
    return scope;
  }

  function legend(target, labels, scope, tagFor) {
    labels.forEach((item,i) => {
      const button = node("button", "legend-item", item.label);
      button.type = "button";
      button.style.setProperty("--mark-color", window.StudyTheme.color(item.color));
      const tags = tagFor(i);
      scope.mark(button,tags);
      scope.bind(button,tags,`${item.label} · highlighted across the figure`);
      target.append(button);
    });
  }

  function bindPlotPoint(element, tags, description, scope) {
    const show=scope.bind(element,tags,description);
    element.addEventListener("keydown",event=>{
      if(event.key==="Enter"||event.key===" "){event.preventDefault();show();}
    });
  }

  function symbol(x,y,color,marker="",size=6) {
    return svgNode(marker.includes("square")?"rect":"circle",{
      ...(marker.includes("square")?{x:x-size,y:y-size,width:2*size,height:2*size}:{cx:x,cy:y,r:size}),
      fill:marker.includes("hollow")?"var(--page)":color,stroke:marker.includes("hollow")?color:"none",
      "stroke-width":1.8,class:"point-symbol",
    });
  }

  function groupedIntervals(target, groups, scope, min=0, max=1, options={}) {
    const W=560, left=options.left??162, right=540, top=20, bottom=top+(options.plotHeight??280), H=bottom+58;
    const x=value=>left+(value-min)/(max-min)*(right-left);
    const centers=options.positions??groups.map((_,i)=>i);
    const span=centers.at(-1)+1;
    const y=i=>top+(centers[i]+.5)/span*(bottom-top);
    const svg=svgNode("svg",{viewBox:`0 0 ${W} ${H}`,class:"interval-svg",role:"group","aria-label":options.label??"Means and 95 percent confidence intervals"});
    target.classList.add("paper-plot");
    if(options.reference){
      const ref=options.reference;
      svg.append(svgNode("rect",{x:x(ref.ci95[0]),y:top,width:x(ref.ci95[1])-x(ref.ci95[0]),height:bottom-top,fill:"#777c84",opacity:.10}));
      svg.append(svgNode("line",{x1:x(ref.estimate),x2:x(ref.estimate),y1:top,y2:bottom,class:"reference-line"}));
    }
    if(min<0&&max>0)svg.append(svgNode("line",{x1:x(0),x2:x(0),y1:top,y2:bottom,class:"plot-zero-line"}));
    svg.append(svgNode("path",{d:`M${left},${top}V${bottom}H${right}`,class:"plot-axis"}));
    const ticks=options.ticks??Array.from({length:5},(_,i)=>min+(max-min)*i/4);
    ticks.forEach(value=>{
      svg.append(svgNode("line",{x1:x(value),x2:x(value),y1:bottom,y2:bottom+5,class:"plot-axis"}));
      svg.append(svgNode("text",{x:x(value),y:bottom+25,"text-anchor":"middle",class:"plot-tick"},Number(value.toFixed(3))));
    });
    groups.forEach((group,i)=>{
      svg.append(svgNode("line",{x1:left-5,x2:left,y1:y(i),y2:y(i),class:"plot-axis"}));
      svg.append(svgNode("text",{x:left-13,y:y(i)+6,"text-anchor":"end",class:"plot-row-label"},group.label));
      const values=[...group.values].sort((a,b)=>(b.ci95[1]-b.ci95[0])-(a.ci95[1]-a.ci95[0]));
      values.forEach(value=>{
        const mark=svgNode("g",{class:"interval-mark"});scope.mark(mark,value.tags);
        const lo=x(value.ci95[0]),hi=x(value.ci95[1]),cy=y(i);
        mark.append(svgNode("path",{d:`M${lo},${cy}H${hi} M${lo+3},${cy-7}Q${lo-4},${cy} ${lo+3},${cy+7} M${hi-3},${cy-7}Q${hi+4},${cy} ${hi-3},${cy+7}`,fill:"none",stroke:value.color,"stroke-width":1.5,class:"interval-ci"}));
        const point=svgNode("g",{class:"interval-point",tabindex:0,role:"button"});
        const px=x(value.mean??value.estimate);
        point.append(svgNode("circle",{cx:px,cy,r:13,fill:"transparent",class:"point-hit"}),symbol(px,cy,value.color,value.marker));
        bindPlotPoint(point,value.tags,`${group.label} · ${value.label}\n${estimate(value)}`,scope);
        mark.append(point);svg.append(mark);
      });
    });
    target.append(svg);
  }

  function entropyFigure(data, summary) {
    const scope = hoverScope(byId("entropy-figure"));
    legend(byId("entropy-legend"),data.sources,scope,i=>({sources:[i]}));
    groupedIntervals(byId("entropy-means"),summary.families.map(family=>({
      label:family.label,
      values:family.values.map((value,i)=>({...value,label:data.sources[i].label,color:data.sources[i].color,tags:{sources:[i]}})),
    })),scope,-.02,1.02,{ticks:[0,.25,.5,.75,1],label:"Mean normalized entropy by puzzle family"});
    const {grid,curves} = data.density;
    const W=560, H=358, left=65, right=540, top=20, bottom=300;
    const ymax=Math.max(...curves.flat())*1.1;
    const x=value=>left+value*(right-left), y=value=>bottom-value/ymax*(bottom-top);
    const svg=svgNode("svg",{viewBox:`0 0 ${W} ${H}`,class:"density-svg","aria-label":"Probability densities of normalized puzzle entropy"});
    for(let i=0;i<=4;i++) {
      const level=i;
      svg.append(svgNode("line",{x1:left-5,x2:left,y1:y(level),y2:y(level),class:"plot-axis"}));
      if(i%2===0)svg.append(svgNode("text",{x:left-10,y:y(level)+6,"text-anchor":"end",class:"plot-tick"},level));
      svg.append(svgNode("line",{x1:x(i/4),x2:x(i/4),y1:bottom,y2:bottom+5,class:"plot-axis"}));
      svg.append(svgNode("text",{x:x(i/4),y:bottom+25,"text-anchor":"middle",class:"plot-tick"},i/4));
    }
    svg.append(svgNode("path",{d:`M${left},${top}V${bottom}H${right}`,class:"plot-axis"}));
    svg.append(svgNode("text",{transform:`translate(18 ${(top+bottom)/2}) rotate(-90)`,class:"plot-axis-label","text-anchor":"middle"},"Probability density"));
    const crosshair=svgNode("line",{y1:top,y2:bottom,class:"density-crosshair",visibility:"hidden"});
    const dot=svgNode("circle",{r:5,stroke:"var(--page)","stroke-width":2,visibility:"hidden"});
    curves.forEach((curve,i)=>{
      const group=svgNode("g");
      scope.mark(group,{sources:[i]});
      const d=curve.map((value,j)=>`${j?"L":"M"}${x(grid[j]).toFixed(3)},${y(value).toFixed(3)}`).join(" ");
      group.append(svgNode("path",{d:d+`L${right},${bottom}L${left},${bottom}Z`,fill:data.sources[i].color,opacity:.12}));
      group.append(svgNode("path",{d,fill:"none",stroke:data.sources[i].color,"stroke-width":2.5,class:"density-line"}));
      const hit=svgNode("path",{d,fill:"none",stroke:"transparent","stroke-width":15,tabindex:0,role:"button",class:"density-hit"});
      const showCurve=scope.bind(hit,{sources:[i]},event=>{
        let index=Math.floor(grid.length/2);
        if(hasPointer(event)) {
          const bounds=svg.getBoundingClientRect();
          const value=Math.max(0,Math.min(1,((event.clientX-bounds.left)*W/bounds.width-left)/(right-left)));
          index=Math.round(value*(grid.length-1));
        }
        crosshair.setAttribute("x1",x(grid[index])); crosshair.setAttribute("x2",x(grid[index]));
        crosshair.setAttribute("visibility","visible");
        dot.setAttribute("cx",x(grid[index])); dot.setAttribute("cy",y(curve[index]));
        dot.setAttribute("fill",window.StudyTheme.color(data.sources[i].color)); dot.setAttribute("visibility","visible");
        return `${data.sources[i].label} · normalized entropy ${number(grid[index])}\nProbability density ${number(curve[index])}`;
      });
      // Enter/Space provides the same midpoint readout as focusing the curve.
      hit.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();showCurve();}});
      group.append(hit); svg.append(group);
    });
    svg.append(crosshair,dot);
    crosshair.setAttribute("visibility","hidden"); dot.setAttribute("visibility","hidden");
    scope.onChange=active=>{ if(!active){crosshair.setAttribute("visibility","hidden");dot.setAttribute("visibility","hidden");} };
    byId("entropy-density").classList.add("paper-plot");
    byId("entropy-density").append(svg);
    byId("entropy-content").hidden=false;
    
  }

  function geometryMap(target, geometry, labels, colors, scope, tagsFor, describe, kind, options={}) {
    const limits=options.limits??(kind==="family"?[[-.66,.72],[-.36,.5]]:[[-.61,.61],[-.36,.36]]);
    const [xmin,xmax]=limits[0], [ymin,ymax]=limits[1];
    const W=560,left=78,right=534,top=30,bottom=top+(right-left)*(ymax-ymin)/(xmax-xmin),H=bottom+65;
    const x=v=>left+(v-xmin)/(xmax-xmin)*(right-left),y=v=>bottom-(v-ymin)/(ymax-ymin)*(bottom-top);
    const positions=geometry.coordinates.map(([a,b])=>[x(a),y(b)]);
    const svg=svgNode("svg",{viewBox:`0 0 ${W} ${H}`,class:"geometry-svg",role:"group","aria-label":options.label??(kind==="family"?"Family geometry":"Source geometry")});
    svg.append(svgNode("line",{x1:left,x2:right,y1:y(0),y2:y(0),class:"plot-zero-line"}),svgNode("line",{x1:x(0),x2:x(0),y1:top,y2:bottom,class:"plot-zero-line"}));
    svg.append(svgNode("path",{d:`M${left},${top}V${bottom}H${right}`,class:"plot-axis"}));
    (options.xticks??(kind==="family"?[-.5,0,.5]:[-.4,0,.4])).forEach(v=>{
      svg.append(svgNode("line",{x1:x(v),x2:x(v),y1:bottom,y2:bottom+5,class:"plot-axis"}),svgNode("text",{x:x(v),y:bottom+25,"text-anchor":"middle",class:"plot-tick"},v));
    });
    (options.yticks??(kind==="family"?[0,.4]:[-.2,0,.2])).forEach(v=>{
      svg.append(svgNode("line",{x1:left-5,x2:left,y1:y(v),y2:y(v),class:"plot-axis"}),svgNode("text",{x:left-12,y:y(v)+6,"text-anchor":"end",class:"plot-tick"},v));
    });
    const prefix=kind==="family"?"CKA":"MDS";
    svg.append(svgNode("text",{x:(left+right)/2,y:bottom+56,"text-anchor":"middle",class:"plot-axis-label"},`${prefix} 1 (${(geometry.shares[0]*100).toFixed(1)}%)`));
    svg.append(svgNode("text",{transform:`translate(20 ${(top+bottom)/2}) rotate(-90)`,"text-anchor":"middle",class:"plot-axis-label"},`${prefix} 2 (${(geometry.shares[1]*100).toFixed(1)}%)`));
    const links=svgNode("g",{class:"map-links"});svg.append(links);
    const offsets=kind==="family"?[[0,27,"middle"],[0,-20,"middle"],[0,-20,"middle"],[14,7,"start"],[-8,23,"end"]]:[[-10,-18,"end"],[12,25,"start"],[0,-22,"middle"],[14,-18,"start"],[14,25,"start"]];
    positions.forEach(([cx,cy],i)=>{
      const point=svgNode("g",{class:"map-point-svg",role:"button",tabindex:0});
      point.append(svgNode("circle",{cx,cy,r:14,fill:"transparent"}),symbol(cx,cy,colors[i],options.markers?.[i]??"",options.sizes?.[i]??9));
      if(labels[i]) {
        const [dx,dy,anchor]=options.offsets?.[i]??offsets[i]??[0,-20,"middle"];
        point.append(svgNode("text",{x:cx+dx,y:cy+dy,"text-anchor":anchor,class:"map-label-svg"},labels[i]));
      }
      scope.mark(point,tagsFor(i));bindPlotPoint(point,tagsFor(i),describe(i),scope);svg.append(point);
    });
    if(options.annotations)options.annotations.forEach(a=>svg.append(svgNode("text",{x:x(a.x),y:y(a.y),"text-anchor":"middle",class:"map-label-svg",fill:a.color},a.label)));
    target.classList.add("paper-plot");target.append(svg);
    return pairs=>{
      links.replaceChildren();
      pairs.forEach(([a,b])=>links.append(svgNode("line",{x1:positions[a][0],y1:positions[a][1],x2:positions[b][0],y2:positions[b][1],class:"map-connector"})));
    };
  }

  function pairwiseHeatmap(target, metric, sources, sourceIndexes, scope) {
    const [min,max]=metric.domain;
    const shade=value=>{
      const normalized=Math.max(0,Math.min(1,(value-min)/(max-min)));
      return metric.direction==="higher"?1-normalized:normalized;
    };
    const blue=value=>{
      const colors=[[8,48,107],[8,81,156],[33,113,181],[66,146,198],[107,174,214],[158,202,225],[198,219,239],[222,235,247],[247,251,255]];
      const index=Math.min(7,Math.floor(value*8)),weight=value*8-index;
      return `rgb(${colors[index].map((v,j)=>Math.round(v*(1-weight)+colors[index+1][j]*weight)).join(",")})`;
    };
    const size=420/sourceIndexes.length,left=108,top=10,bottom=430;
    const heatmap=svgNode("svg",{viewBox:"0 0 642 485",class:"heatmap-svg",role:"group","aria-label":`${metric.label}: pairwise means with 95 percent intervals`});
    sourceIndexes.forEach((a,row)=>{
      [[left-12,top+(row+.5)*size+6,"end"],[left+(row+.5)*size,bottom+29,"middle"]].forEach(([x,y,anchor])=>{
        const heading=svgNode("text",{x,y,"text-anchor":anchor,class:"matrix-label",tabindex:0,role:"button"},sources[a]);
        scope.mark(heading,{sources:[a]});bindPlotPoint(heading,{sources:[a]},`${sources[a]} · comparisons highlighted`,scope);heatmap.append(heading);
      });
      sourceIndexes.forEach((b,column)=>{
        const cell=svgNode("g",{class:"tv-cell",...(a!==b?{tabindex:0,role:"button"}:{})});
        const pair=metric.pairs.find(p=>p.a===Math.min(a,b)&&p.b===Math.max(a,b));
        const value=a===b?(metric.direction==="higher"?1:0):pair.mean;
        const intensity=shade(value),cx=left+(column+.5)*size,cy=top+(row+.5)*size;
        const ink=a===b||(metric.id!=="total_variation"&&intensity<.38)?"white":"#111";
        cell.append(svgNode("rect",{x:left+column*size,y:top+row*size,width:size,height:size,fill:blue(intensity),stroke:"white","stroke-width":3}));
        cell.append(svgNode("text",{x:cx,y:cy+(a===b?7:-4),"text-anchor":"middle",style:`fill:${ink}`,class:"matrix-mean"},value.toFixed(2)));
        if(a!==b){
          cell.append(svgNode("text",{x:cx,y:cy+22,"text-anchor":"middle",style:`fill:${ink}`,class:"matrix-ci"},`[${pair.ci95.map(v=>v.toFixed(2)).join(",")}]`));
          const tags={pair:pairKey(pair),sources:[a,b]};scope.mark(cell,tags);
          bindPlotPoint(cell,tags,`${sources[pair.a]} – ${sources[pair.b]}\n${metric.label} ${estimate(pair)}`,scope);
        }else cell.setAttribute("aria-label",`${sources[a]} versus itself: exactly ${value}`);
        heatmap.append(cell);
      });
    });
    for(let i=0;i<100;i++)heatmap.append(svgNode("rect",{x:552,y:top+(99-i)*420/100,width:16,height:420/100+.5,fill:blue(shade(min+(max-min)*i/99))}));
    [min,(min+max)/2,max].forEach(v=>heatmap.append(svgNode("text",{x:574,y:bottom-(v-min)/(max-min)*420+6,class:"plot-tick"},v)));
    heatmap.append(svgNode("text",{transform:"translate(635 220) rotate(-90)","text-anchor":"middle",class:"plot-axis-label"},metric.id==="total_variation"?"Mean TV":metric.label));
    target.classList.add("paper-plot");target.append(heatmap);
  }

  function metricMapOptions(geometry, label) {
    // Leave room for labels; preserve the same unit length on both axes.
    let x=Math.max(.25,...geometry.coordinates.map(p=>Math.abs(p[0])*1.45));
    let y=Math.max(.18,...geometry.coordinates.map(p=>Math.abs(p[1])*1.45));
    x=Math.max(x,1.65*y);y=Math.max(y,x/2.2);
    const ticks=limit=>{
      const step=limit>.6?.5:limit>.3?.25:.1;
      const count=Math.floor(limit/step);
      return Array.from({length:2*count+1},(_,i)=>Number(((i-count)*step).toFixed(2)));
    };
    return {limits:[[-x,x],[-y,y]],xticks:ticks(x),yticks:ticks(y),label};
  }

  function tvFigure(data) {
    const tv=data.tv, scope=hoverScope(byId("tv-content")), matrix=byId("tv-matrix");
    const pairName=pair=>`${tv.sources[pair.a]} – ${tv.sources[pair.b]}`;
    pairwiseHeatmap(matrix,{...tv,id:"total_variation",label:"Total variation",domain:[0,1],direction:"lower"},tv.sources,[0,1,2,3,4],scope);
    legend(byId("tv-family-legend"),data.families.map((label,i)=>({label,color:familyColors[i]})),scope,i=>({family:i}));
    groupedIntervals(byId("tv-family-comparisons"),tv.pairs.map(pair=>({
      label:pairName(pair),values:pair.families.map((value,family)=>({...value,label:data.families[family],color:familyColors[family],tags:{pair:pairKey(pair),family,sources:[pair.a,pair.b]}})),
    })),scope,0,1,{left:210,plotHeight:440,positions:[0,1,2,3,4.6,5.6,6.6,8.2,9.2,10.2],label:"TV by source pair and puzzle family"});
    const drawLinks=geometryMap(byId("source-map"),tv.geometry.source,tv.sources,["#777C84",...data.sources.map(s=>s.color)],scope,i=>({sources:[i]}),i=>{
      const comparisons=tv.pairs.filter(pair=>pair.a===i||pair.b===i);
      return `${tv.sources[i]} · exact mean TV to other sources\n`+comparisons.map(pair=>`${tv.sources[pair.a===i?pair.b:pair.a]}: ${estimate(pair)}`).join("\n");
    },"source");
    geometryMap(byId("family-map"),tv.geometry.family,data.families,familyColors,scope,i=>({family:i}),i=>`${data.families[i]} · family geometry\nIts TV comparisons are highlighted above.\nCKA map coordinates: ${tv.geometry.family.coordinates[i].map(number).join(", ")}`,"family");
    scope.onChange=active=>{
      if(active?.pair!==undefined){const pair=tv.pairs.find(p=>pairKey(p)===active.pair);drawLinks([[pair.a,pair.b]]);}
      else if(active?.sources?.length===1){const i=active.sources[0];drawLinks(tv.pairs.filter(p=>p.a===i||p.b===i).map(p=>[p.a,p.b]));}
      else drawLinks([]);
    };
    byId("tv-content").hidden=false;
    return scope;
  }

  function effectFigure(data) {
    const scope=hoverScope(byId("effects-figure"));
    legend(byId("effects-legend"),data.sources,scope,i=>({sources:[i]}));
    const names={cue:"Highlighting",pair_related:"Related context",transfer:"Strategy primer",spatial:"Spatial reflection",formulation:"Number ordering",pair_related_mi:"Related",pair_control_mi:"Unrelated"};
    ["tv","gain","mi"].forEach(metric=>{
      const rows=data.effects[metric],keys=[...new Set(rows.map(row=>row.module||row.metric))];
      const groups=keys.map(key=>({label:names[key],values:data.sources.map((source,i)=>({
        ...rows.find(row=>(row.module||row.metric)===key&&row.source===source.name),
        label:source.label,color:source.color,tags:{sources:[i]},
      }))}));
      const settings={
        tv:{min:0,max:1,left:206,plotHeight:520,positions:[0,1,2,3.45,4.45],ticks:[0,.5,1]},
        gain:{min:-.85,max:1.05,left:184,plotHeight:215,ticks:[-.5,0,.5,1]},
        mi:{min:0,max:.7,left:184,plotHeight:170,ticks:[0,.3,.6]},
      }[metric];
      groupedIntervals(byId("effect-"+metric),groups,scope,settings.min,settings.max,settings);
    });
    const contrasts=data.context_contrasts;
    const contrastValue=row=>({mean:row.estimate_bits,ci95:[row.ci95_low,row.ci95_high]});
    groupedIntervals(byId("contrast-context"),data.sources.map((source,i)=>({label:source.label,values:[{
      ...contrastValue(contrasts.find(row=>row.comparison==="Related minus unrelated"&&row.source===source.label)),
      label:"Related minus unrelated MI (bits)",color:source.color,tags:{sources:[i]},
    }]})),scope,-.1,.27,{ticks:[-.1,0,.1,.2],plotHeight:230});
    groupedIntervals(byId("contrast-source"),["Related","Unrelated"].map(context=>({label:context,values:data.sources.slice(1).map((source,i)=>({
      ...contrastValue(contrasts.find(row=>row.comparison==="Human minus model"&&row.source===source.label&&row.context===context)),
      label:`Human minus ${source.label} MI (bits)`,color:source.color,tags:{sources:[i+1]},
    }))})),scope,-.03,.65,{ticks:[0,.2,.4,.6],plotHeight:230});
    byId("context-contrasts").hidden=false;
    byId("effects-content").hidden=false;
  }

  function difficultyFigure(data) {
    const scope=hoverScope(byId("difficulty-figure")), difficulty=data.difficulty;
    const familyKeys=["arithmetic24","maze","grid_placement","minesweeper_lite","mini_sudoku"];
    legend(byId("difficulty-legend"),data.sources,scope,i=>({sources:[i]}));
    groupedIntervals(byId("difficulty-means"),data.families.map((family,f)=>({label:family,values:data.sources.map((source,i)=>({
      ...difficulty.family_means.find(row=>row.family===familyKeys[f]&&row.source===source.name),
      label:source.label+" · centered log effort",color:source.color,tags:{sources:[i],family:f},
    }))})),scope,-1.75,1.5,{ticks:[-1,0,1]});
    groupedIntervals(byId("difficulty-correlations"),data.sources.slice(1).map((source,i)=>{
      const row=difficulty.correlations.find(row=>row.source===source.name);
      return {label:`Human – ${source.label}`,values:[
        {...row.mean,label:"Mean within-family Pearson r",color:source.color,tags:{sources:[i+1]}},
        {...row.pooled,label:"Pooled Pearson r",color:source.color,marker:"hollow",tags:{sources:[i+1]}},
      ]};
    }),scope,-1,1,{left:195,ticks:[-1,-.5,0,.5,1]});
    window.drawDifficultyMatrix(byId("difficulty-scatterplots"),data,scope,{svgNode,number,familyColors});
    byId("difficulty-content").hidden=false;
  }

  function distributionFigure(data) {
    const choices=[{
      id:"total_variation",label:"Total variation",direction:"lower",domain:[0,1],
      description:"0 means identical distributions; 1 means no overlap.",
      view:byId("tv-content"),scope:tvFigure(data),
    }];
    const pairName=pair=>`${data.tv.sources[pair.a]} – ${data.tv.sources[pair.b]}`;
    data.alternative_measures.forEach(metric=>{
      const view=node("div","reactive-content");
      view.id=`measure-${metric.id}`;
      view.hidden=true;
      byId("alternative-views").append(view);
      const scope=hoverScope(view);
      const direction=metric.direction==="lower"?"Lower = closer":"Higher = more similar";
      view.append(node("p","chart-scroll-note","Wide charts and tables scroll horizontally."),
        node("p","hover-instruction","Hover or tap a source pair to see its mean and 95% CI and highlight its family comparisons and map points. Hover over a family to follow its pattern."));
      const panels=node("div","figure-panels two-panels");
      view.append(panels);
      const addPanel=(letter,title)=>{
        const panel=node("section","chart-panel"),heading=node("h4"),plot=node("div");
        heading.append(node("span","panel-letter",letter),document.createTextNode(" "+title));
        panel.append(heading,plot);
        panels.append(panel);
        return {panel,plot};
      };
      const sourceIndexes=metric.geometry.source.source_indexes;
      const overall=addPanel("a","Overall pairwise comparisons");
      pairwiseHeatmap(overall.plot,metric,data.tv.sources,sourceIndexes,scope);
      overall.panel.append(node("p","panel-note",`Equal-family means and 95% CIs. ${direction}. Darker cells indicate more similar distributions.`));
      const families=addPanel("b","Comparisons within each family");
      groupedIntervals(families.plot,metric.pairs.map(pair=>({label:pairName(pair),
        values:pair.families.map((value,family)=>({...value,label:data.families[family],color:familyColors[family],tags:{pair:pairKey(pair),family,sources:[pair.a,pair.b]}})),
      })),scope,...metric.domain,{left:210,plotHeight:440,positions:[0,1,2,3.7,4.7,5.7],
        ticks:metric.domain[0]<0?[-1,-.5,0,.5,1]:[0,.25,.5,.75,1],
        label:`${metric.label}: within-family means and 95 percent confidence intervals. ${direction}.`});
      families.panel.append(node("p","axis-description",`${metric.label} · ${direction.toLowerCase()}`));

      const source=addPanel("c","Source geometry");
      const drawLinks=geometryMap(source.plot,metric.geometry.source,
        sourceIndexes.map(i=>data.tv.sources[i]),sourceIndexes.map(i=>data.sources[i-1].color),scope,i=>({sources:[sourceIndexes[i]]}),i=>{
          const index=sourceIndexes[i],comparisons=metric.pairs.filter(pair=>pair.a===index||pair.b===index);
          return `${data.tv.sources[index]} · ${metric.label}\n`+comparisons.map(pair=>`${data.tv.sources[pair.a===index?pair.b:pair.a]}: ${estimate(pair)}`).join("\n");
        },"source",{...metricMapOptions(metric.geometry.source,`${metric.label}: source geometry`),
          offsets:[[0,27,"middle"],[0,-22,"middle"],[-14,-18,"end"],[14,25,"start"]]});
      const sourceNote=metric.direction==="lower"
        ?`MDS places sources so that distances approximate their mean ${metric.label.toLowerCase()}.`
        :"MDS converts mean similarity s to dissimilarity √(2 × (1 − s)). Map distances approximate that dissimilarity; closer points are more similar.";
      source.panel.append(node("p","panel-note",`${sourceNote} Hover or tap a source to see its exact pairwise values.`));
      const family=addPanel("d","Family geometry");
      const familyOffsets={
        hellinger_distance:[[0,-20,"middle"],[0,-20,"middle"],[0,-20,"middle"],[14,7,"start"],[-8,23,"end"]],
        log_density_correlation:[[0,27,"middle"],[0,-20,"middle"],[0,-20,"middle"],[14,7,"start"],[14,7,"start"]],
        tie_aware_modal_agreement:[[0,27,"middle"],[0,-20,"middle"],[0,27,"middle"],[-14,7,"end"],[-8,23,"end"]],
      };
      geometryMap(family.plot,metric.geometry.family,data.families,familyColors,scope,i=>({family:i}),i=>
        `${data.families[i]} · ${metric.label}\nIts family comparisons are highlighted above.\nCKA map coordinates: ${metric.geometry.family.coordinates[i].map(number).join(", ")}`,
        "family",{...metricMapOptions(metric.geometry.family,`${metric.label}: family geometry`),offsets:familyOffsets[metric.id]});
      family.panel.append(node("p","panel-note","Families lie near one another when this measure gives similar patterns of differences among the four sources, independent of overall scale. CKA compares the full source geometries before they are reduced to two dimensions."));
      scope.onChange=active=>{
        let pairs=[];
        if(active?.pair!==undefined)pairs=metric.pairs.filter(pair=>pairKey(pair)===active.pair);
        else if(active?.sources?.length===1)pairs=metric.pairs.filter(pair=>pair.a===active.sources[0]||pair.b===active.sources[0]);
        drawLinks(pairs.map(pair=>[sourceIndexes.indexOf(pair.a),sourceIndexes.indexOf(pair.b)]));
      };
      const familyLegend=node("div","figure-legend family-legend");
      familyLegend.setAttribute("aria-label","Puzzle families");
      legend(familyLegend,data.families.map((label,i)=>({label,color:familyColors[i]})),scope,i=>({family:i}));
      view.append(familyLegend);

      const details=node("details","alternative-details");
      details.append(node("summary","","View estimates by puzzle family"),
        node("p","panel-note","Each cell gives the estimate and pointwise 95% CI. The Mean column weights all five families equally."));
      const wrapper=node("div","alternative-table-scroll");
      wrapper.tabIndex=0;
      wrapper.setAttribute("role","region");
      wrapper.setAttribute("aria-label",`${metric.label} by puzzle family; scroll horizontally on small screens`);
      const table=node("table","alternative-table");
      table.append(node("caption","",`${metric.label} · ${direction.toLowerCase()}`));
      const head=node("thead"), header=node("tr");
      ["Comparison","Mean",...data.families].forEach(label=>{
        const cell=node("th","",label);cell.scope="col";header.append(cell);
      });
      head.append(header);table.append(head);
      const body=node("tbody");
      metric.pairs.forEach(pair=>{
        const row=node("tr");
        const label=node("th","",pairName(pair));label.scope="row";row.append(label);
        [pair,...pair.families].forEach(value=>{
          const cell=node("td");
          cell.append(node("span","metric-estimate",value.mean.toFixed(2)),
            node("span","metric-ci",`[${value.ci95.map(v=>v.toFixed(2)).join(", ")}]`));
          row.append(cell);
        });
        scope.mark(row,{pair:pairKey(pair),sources:[pair.a,pair.b]});body.append(row);
      });
      table.append(body);wrapper.append(table);details.append(wrapper);view.append(details);
      view.append(node("p","panel-note","All four alternatives use the same six human–model and model–model pairs. Uniform is omitted: its log-density correlation is undefined, and its modal agreement is always 1 because every solution ties for most frequent."));
      if(metric.id==="log_density_correlation")view.append(node("p","panel-note","Log-density correlation describes which solutions are favored; it cannot detect a pure change in concentration."));
      choices.push({...metric,view,scope});
    });
    const select=byId("distribution-measure");
    choices.forEach(metric=>{
      const option=node("option","",metric.label);option.value=metric.id;select.append(option);
    });
    const showMeasure=()=>{
      const selected=choices.find(metric=>metric.id===select.value);
      choices.forEach(metric=>{metric.scope.clear();metric.view.hidden=metric!==selected;});
      const direction=selected.direction==="lower"?"Lower means closer":"Higher means more similar";
      byId("measure-description").textContent=`${direction} (${selected.domain.join(" to ")}). ${selected.description}`;
    };
    select.addEventListener("change",showMeasure);
    showMeasure();
    byId("measure-controls").hidden=false;
  }

  function promptingFigure(data) {
    const scope=hoverScope(byId("prompting-figure")), models=data.sources.slice(1), prompting=data.prompting;
    legend(byId("prompting-legend"),data.sources,scope,i=>({sources:[i]}));
    const marker=condition=>[condition.startsWith("Medium")?"square":"",condition.includes("persona")?"hollow":""].filter(Boolean).join(" ");
    ["levels","shifts"].forEach(kind=>{
      const rows=prompting[kind], conditions=[...new Set(rows.map(row=>row.condition))];
      groupedIntervals(byId("prompting-"+kind),conditions.map(condition=>({label:condition,values:models.map((source,i)=>({
        ...rows.find(row=>row.provider===source.name&&row.condition===condition),
        label:source.label+(kind==="levels"?" · TV from Human":" · TV from low/plain"),color:source.color,marker:marker(condition),tags:{sources:[i+1]},
      }))})),scope,kind==="levels"?.25:0,kind==="levels"?.6:.21,{
        left:176,plotHeight:kind==="levels"?235:185,ticks:kind==="levels"?[.3,.4,.5,.6]:[0,.05,.1,.15,.2],
        ...(kind==="levels"?{reference:prompting.uniform_reference}:{}),
      });
    });
    const geometry=prompting.geometry;
    const sourceIndices=geometry.source_order.map((_,i)=>i===0?-1:i===1?0:1+Math.floor((i-2)/4));
    const conditions=["Low plain","Low persona","Medium plain","Medium persona"];
    const markers=geometry.source_order.map((_,i)=>i<2?"":marker(conditions[(i-2)%4]));
    const colors=sourceIndices.map(i=>i<0?"#777C84":data.sources[i].color);
    const labels=geometry.source_order.map((_,i)=>i===0?"Uniform":i===1?"Human":"");
    const annotations=models.map((source,i)=>{
      const points=geometry.coordinates.slice(2+i*4,6+i*4);
      return {x:points.reduce((v,p)=>v+p[0],0)/4,y:Math.max(...points.map(p=>p[1]))+.055,label:source.label,color:source.color};
    });
    geometryMap(byId("prompting-geometry"),geometry,labels,colors,scope,i=>({sources:[sourceIndices[i]]}),i=>{
      const source=sourceIndices[i];
      const name=i===0?"Uniform":i===1?"Human":`${data.sources[source].label} · ${conditions[(i-2)%4]}`;
      const row=i===0?prompting.uniform_reference:i>1?prompting.levels.find(row=>row.provider===data.sources[source].name&&row.condition===conditions[(i-2)%4]):null;
      return `${name}\nMDS coordinates: ${geometry.coordinates[i].map(number).join(", ")}`+(row?`\nTV from Human: ${estimate(row)}`:"");
    },"source",{limits:[[-.58,.51],[-.32,.38]],xticks:[-.4,0,.4],yticks:[-.2,0,.2],markers,sizes:sourceIndices.map((_,i)=>i<2?9:6),annotations,
      offsets:[[0,26,"middle"],[0,26,"middle"]],label:"Source geometry across prompting and effort conditions"});
    byId("uniform-reference").textContent=`Dashed line and gray band: uniform versus Human, ${estimate(prompting.uniform_reference)}.`;
    byId("prompting-content").hidden=false;
  }

  async function initialize() {
    const renderers={entropy:entropyFigure,tv:distributionFigure,effects:effectFigure,difficulty:difficultyFigure,prompting:promptingFigure};
    const fail=(name,error)=>{
      byId(name+"-status").textContent="This figure could not load. Please reload the page, or download the values above.";
      byId(name+"-status").hidden=false;
      console.error(`${name} figure unavailable`,error);
    };
    try {
      const responses=await Promise.all([fetch("assets/figure-data.json?v=metric-geometry-1"),fetch("assets/entropy-summary.json")]);
      if(responses.some(response=>!response.ok))throw new Error("Figure data unavailable");
      const [data,summary]=await Promise.all(responses.map(response=>response.json()));
      Object.entries(renderers).forEach(([name,render])=>{
        try{render(data,summary);byId(name+"-status").hidden=true;}catch(error){fail(name,error);}
      });
    } catch(error) {
      Object.keys(renderers).forEach(name=>fail(name,error));
    }
  }
  initialize();
})();
