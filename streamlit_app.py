import streamlit as st
import pandas as pd
import networkx as nx
import folium
from folium import PolyLine, Marker
from streamlit_folium import st_folium
from io import BytesIO

st.set_page_config(page_title="Најкраћа и алтернативне руте", layout="wide")
st.title("🛤️ Израчунавање најкраће руте + алтернативе + мапа")

@st.cache_data
def load_data():
    return pd.read_excel('Izjava o mreži 2025.xlsx')

df = load_data()

junctions = df['Назив службеног места'][df['Назив службеног места'].duplicated(keep=False)].unique()
key_junctions = ["НИШ","ВЕЛИКА ПЛАНА","МАЛА КРСНА","РЕСНИК","БЕОГРАД ЦЕНТАР","НОВИ САД","ПАНЧЕВО ГЛАВНА","ОРЛОВАТ"]

grouped = df.groupby('Број пруге')
main_lines, secondary_lines = [], []
for line_number, group in grouped:
    try: number=int(line_number)
    except: number=None
    stations=group.sort_values('Редни број службеног места на прузи')
    if (number and 100<=number<=150) or len(stations)>6:
        main_lines.append((line_number,stations))
    else: secondary_lines.append((line_number,stations))

graph_full = nx.Graph()
def build_graph(lines, G):
    for _, stations in lines:
        prev=None
        for _, row in stations.iterrows():
            curr=row['Назив службеног места']
            if prev:
                dist=row['Километарска удаљеност']/1000
                G.add_edge(prev,curr,weight=dist)
            prev=curr

build_graph(main_lines+secondary_lines, graph_full)

def find_shortest_route_multi(G, start, vias, end):
    try:
        path, used, current, G_current = [], set(), start, G.copy()
        for via in vias+[end]:
            seg=nx.shortest_path(G_current,current,via,weight='weight')
            for node in seg:
                if node!=via: used.add(node)
            G_current.remove_nodes_from([n for n in used if n!=via])
            path+=seg if not path else seg[1:]
            current=via
        dist=sum(G[u][v]['weight'] for u,v in zip(path[:-1],path[1:]))
        return path, dist
    except: return None, None

def find_alternative_routes_obilasci(G, base_path, n, allow_return=False):
    alternatives=[]
    tried=set()
    for key in key_junctions:
        if key in base_path and key not in tried:
            tried.add(key)
            G_tmp=G.copy()
            G_tmp.remove_node(key)
            try:
                alt_path,dist=find_shortest_route_multi(G_tmp,base_path[0],[],base_path[-1])
                if alt_path and alt_path!=base_path:
                    alt_junc = set(x for x in alt_path if x in junctions)
                    base_junc = set(x for x in base_path if x in junctions)
                    diff = len(alt_junc ^ base_junc)
                    if diff>=1:
                        alternatives.append((alt_path,dist))
            except: continue
        if len(alternatives)>=n: break
    return alternatives

all_stations=sorted(df['Назив службеног места'].unique())

start_idx = all_stations.index("ДРЖАВНА ГРАНИЦА ПРЕШЕВО") if "ДРЖАВНА ГРАНИЦА ПРЕШЕВО" in all_stations else 0
end_idx = all_stations.index("ДРЖАВНА ГРАНИЦА СУБОТИЦА") if "ДРЖАВНА ГРАНИЦА СУБОТИЦА" in all_stations else 0

col1,col2=st.columns([1.3,2])

with col1:
    start=st.selectbox("🚩 Почетна станица", all_stations, index=start_idx)
    end=st.selectbox("🏁 Крајња станица", all_stations, index=end_idx)
    vias=st.multiselect("🔀 Међустанице (опционално, редом)", all_stations)
    exclude_nodes=st.multiselect("🚫 Изабери службена места које рута не сме да користи (опционално)", all_stations)
    num_alternatives=st.selectbox("➕ Број алтернативних рута",[0,1,3,5,10],index=0)
    allow_return=st.checkbox("🔄 Дозволи повратне руте",value=False)
    show_all=st.checkbox("👀 Прикажи све станице на рути",value=False)

    b1,b2=st.columns(2)
    with b1:
        if st.button("Израчунај руту"):
            G_tmp=graph_full.copy()
            if exclude_nodes: G_tmp.remove_nodes_from(exclude_nodes)
            path, dist=find_shortest_route_multi(G_tmp,start,vias,end)
            if path:
                st.session_state["base_path"]=path
                st.session_state["base_distance"]=dist
                st.session_state["alt_paths"]=find_alternative_routes_obilasci(G_tmp,path,num_alternatives,allow_return)
                st.session_state["show_map"]=False
            else:
                st.session_state["base_path"]=None
                st.session_state["alt_paths"]=[]
                st.session_state["show_map"]=False
                st.error("⚠️ Није пронађена рута.")
    with b2:
        if st.button("Прикажи мапу"):
            st.session_state["show_map"]=True

    dfs = {}
    if "base_path" in st.session_state and st.session_state["base_path"]:
        st.success(f"✅ Најкраћа рута пронађена! {st.session_state['base_distance']:.2f} km")
        path=st.session_state["base_path"]
        rows,cum=[],0.0
        for i,name in enumerate(path):
            uic_row=df[df['Назив службеног места']==name].iloc[0]
            uic="" if pd.isna(uic_row['Шифра службеног места - UIC']) else str(int(uic_row['Шифра службеног места - UIC']))
            if i==0:
                rows.append({"#":i+1,"UIC":uic,"Службено место":name,"Растојање између сл. места (км)":"0.00","Кумулативно (км)":"0.00"})
            else:
                d=graph_full[path[i-1]][name]['weight']
                cum+=d
                rows.append({"#":i+1,"UIC":uic,"Службено место":name,"Растојање између сл. места (км)":f"{d:.2f}","Кумулативно (км)":f"{cum:.2f}"})
        df_main=pd.DataFrame(rows)
        if not show_all:
            filtered=[r for r in rows if r["Службено место"] in junctions or r["#"]=="1" or r["Службено место"]==path[-1]]
            df_main=pd.DataFrame(filtered)[["#","UIC","Службено место","Кумулативно (км)"]]
        dfs["Најкраћа рута"]=df_main
        st.markdown("### 🛤️ Најкраћа рута")
        st.table(df_main)

        if st.session_state["alt_paths"]:
            for idx,(alt,dist) in enumerate(st.session_state["alt_paths"]):
                rows_alt,cum_alt=[],0.0
                for i,name in enumerate(alt):
                    uic_row=df[df['Назив службеног места']==name].iloc[0]
                    uic="" if pd.isna(uic_row['Шифра службеног места - UIC']) else str(int(uic_row['Шифра службеног места - UIC']))
                    if i==0:
                        rows_alt.append({"#":i+1,"UIC":uic,"Службено место":name,"Растојање између сл. места (км)":"0.00","Кумулативно (км)":"0.00"})
                    else:
                        d=graph_full[alt[i-1]][name]['weight']
                        cum_alt+=d
                        rows_alt.append({"#":i+1,"UIC":uic,"Службено место":name,"Растојање између сл. места (км)":f"{d:.2f}","Кумулативно (км)":f"{cum_alt:.2f}"})
                df_alt=pd.DataFrame(rows_alt)
                if not show_all:
                    filtered=[r for r in rows_alt if r["Службено место"] in junctions or r["#"]=="1" or r["Службено место"]==alt[-1]]
                    df_alt=pd.DataFrame(filtered)[["#","UIC","Службено место","Кумулативно (км)"]]
                dfs[f"Алт. рута #{idx+1}"]=df_alt
                st.markdown(f"### 🛤️ Алт. рута #{idx+1} – {dist:.2f} km")
                st.table(df_alt)

        if st.button("📥 Експорт у Excel"):
            output=BytesIO()
            with pd.ExcelWriter(output,engine='xlsxwriter') as writer:
                for sheet,dfx in dfs.items():
                    dfx.to_excel(writer,index=False,sheet_name=sheet)
            output.seek(0)
            st.download_button("⬇️ Преузми Excel", data=output, file_name="rute.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with col2:
    if st.session_state.get("show_map",False) and "base_path" in st.session_state:
        coords=[]
        all_coords=[]
        path=st.session_state["base_path"]
        for name in path:
            row=df[df['Назив службеног места']==name].iloc[0]
            lat,lon=row['Координата1'],row['Координата2']
            if lat!=0 and lon!=0:
                coords.append((lat,lon))
                all_coords.append((lat,lon))
        colors=["green","purple","orange","cadetblue","darkred"]
        for i,(p,d) in enumerate(st.session_state["alt_paths"]):
            for name in p:
                row=df[df['Назив службеног места']==name].iloc[0]
                lat,lon=row['Координата1'],row['Координата2']
                if lat!=0 and lon!=0:
                    all_coords.append((lat,lon))
        m=folium.Map(tiles='CartoDB positron')
        m.fit_bounds(all_coords)
        PolyLine(coords,color="red",weight=5,tooltip="Најкраћа рута").add_to(m)

        # Заставице и чиоде
        start_row=df[df['Назив службеног места']==path[0]].iloc[0]
        Marker((start_row['Координата1'],start_row['Координата2']),tooltip=f"Почетак: {path[0]}",
               icon=folium.Icon(color='green',icon='flag')).add_to(m)
        end_row=df[df['Назив службеног места']==path[-1]].iloc[0]
        Marker((end_row['Координата1'],end_row['Координата2']),tooltip=f"Крај: {path[-1]}",
               icon=folium.Icon(color='red',icon='flag')).add_to(m)
        for via in vias:
            via_row=df[df['Назив службеног места']==via].iloc[0]
            Marker((via_row['Координата1'],via_row['Координата2']),tooltip=f"Међустаница: {via}",
                   icon=folium.Icon(color='blue',icon='flag')).add_to(m)
        for n in path:
            if n in junctions:
                row=df[df['Назив службеног места']==n].iloc[0]
                Marker((row['Координата1'],row['Координата2']),
                       icon=folium.Icon(color='blue',icon='star'),
                       tooltip=f"Чворна: {n}").add_to(m)
        # Алтернативе
        for i,(p,d) in enumerate(st.session_state["alt_paths"]):
            coords_alt=[]
            for name in p:
                row=df[df['Назив службеног места']==name].iloc[0]
                lat,lon=row['Координата1'],row['Координата2']
                if lat!=0 and lon!=0:
                    coords_alt.append((lat,lon))
            if coords_alt:
                PolyLine(coords_alt,color=colors[i%len(colors)],weight=3,tooltip=f"Алт. рута #{i+1} ({d:.1f} km)").add_to(m)
        st_folium(m,width=700,height=600)
    else:
        m=folium.Map(location=[44,21],zoom_start=7,tiles='CartoDB positron')
        st_folium(m,width=700,height=600)
