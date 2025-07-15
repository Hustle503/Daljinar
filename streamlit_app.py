import streamlit as st
import pandas as pd
import networkx as nx
import folium
from folium import PolyLine, Marker
from streamlit_folium import st_folium

st.set_page_config(page_title="Најкраћа рута + мапа", layout="wide")
st.title("🛤️ Израчунавање најкраће руте + мапа + алтернативе")

@st.cache_data
def load_data():
    return pd.read_excel('Izjava o mreži 2025.xlsx')

df = load_data()

# Чворне станице = дупликати
junctions = df['Назив службеног места'][df['Назив службеног места'].duplicated(keep=False)].unique()

# Груписање пруга
grouped = df.groupby('Број пруге')
main_lines = []
secondary_lines = []

for line_number, group in grouped:
    try:
        number = int(line_number)
    except ValueError:
        number = None
    stations = group.sort_values('Редни број службеног места на прузи')
    if (number is not None and 100 <= number <= 150) or len(stations) > 6:
        main_lines.append((line_number, stations))
    else:
        secondary_lines.append((line_number, stations))

# Графови
graph_main = nx.Graph()
graph_full = nx.Graph()

def build_graph(lines, G):
    for _, stations in lines:
        prev = None
        for _, row in stations.iterrows():
            curr = row['Назив службеног места']
            if prev:
                dist = row['Километарска удаљеност'] / 1000
                G.add_edge(prev, curr, weight=dist)
            prev = curr

build_graph(main_lines, graph_main)
build_graph(main_lines + secondary_lines, graph_full)

# Најкраћа рута са више међустаница
def find_shortest_route_multi(G, start, vias, end):
    try:
        full_path = []
        used_nodes = set()
        current = start
        G_current = G.copy()
        for via in vias + [end]:
            path_segment = nx.shortest_path(G_current, current, via, weight='weight')
            for node in path_segment:
                if node != via:
                    used_nodes.add(node)
            G_current.remove_nodes_from([n for n in used_nodes if n != via])
            full_path += path_segment if not full_path else path_segment[1:]
            current = via
        dist = sum(G[u][v]['weight'] for u,v in zip(full_path[:-1], full_path[1:]))
        return full_path, dist
    except:
        return None, None

# Алтернативне руте
def find_alternative_routes(G, base_path, n, allow_return=False):
    if not base_path:
        return []
    try:
        all_paths = list(nx.all_simple_paths(G, base_path[0], base_path[-1], cutoff=20))
        scored = []
        for p in all_paths:
            if not allow_return:
                edges = set((min(u,v), max(u,v)) for u,v in zip(p[:-1], p[1:]))
                if len(edges) < len(p)-1:
                    continue
            dist = sum(G[u][v]['weight'] for u,v in zip(p[:-1], p[1:]))
            scored.append((p, dist))
        scored.sort(key=lambda x: x[1])
        unique = []
        for cand, dist in scored:
            if cand == base_path:
                continue
            diff = len(set(cand) ^ set(base_path))
            if diff >= 5:
                unique.append((cand, dist))
            if len(unique) >= n:
                break
        return unique
    except:
        return []

# UI
all_stations = sorted(df['Назив службеног места'].unique())
num_alternatives = st.selectbox("➕ Број алтернативних рута", [0,1,3,5,10], index=0)

col1, col2 = st.columns([1,2])

with col1:
    start = st.selectbox("🚩 Почетна станица", all_stations)
    end = st.selectbox("🏁 Крајња станица", all_stations)
    vias = st.multiselect("🔀 Међустанице (опционално, редом)", all_stations)
    allow_return = st.checkbox("🔄 Дозволи повратне руте (са дуплирањем)", value=False)
    show_all = st.checkbox("👀 Прикажи све станице на рути", value=False)

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("Израчунај руту"):
            path, dist = find_shortest_route_multi(graph_full, start, vias, end)
            if path:
                st.session_state["base_path"] = path
                st.session_state["base_distance"] = dist
                st.session_state["alt_paths"] = find_alternative_routes(graph_main, path, num_alternatives, allow_return)
                st.session_state["show_map"] = False
            else:
                st.session_state["base_path"]=None
                st.session_state["alt_paths"]=[]
                st.session_state["show_map"]=False
                st.error("⚠️ Није пронађена рута.")
    with btn2:
        if st.button("Прикажи мапу"):
            st.session_state["show_map"]=True

    if "base_path" in st.session_state and st.session_state["base_path"]:
        st.success(f"✅ Најкраћа рута пронађена! {st.session_state['base_distance']:.2f} km")

        path = st.session_state["base_path"]

        if show_all:
            # Пуна табела: све станице
            rows = []
            cumulative = 0.0
            for i, name in enumerate(path):
                uic_row = df[df['Назив службеног места']==name].iloc[0]
                uic_value = uic_row['Шифра службеног места - UIC']
                uic = "" if pd.isna(uic_value) else str(int(uic_value))
                if i == 0:
                    rows.append({
                        "Редослед": i+1,
                        "UIC": uic,
                        "Службено место": name,
                        "Растојање између сл. места (км)": "0.00",
                        "Кумулативно растојање (км)": "0.00"
                    })
                else:
                    distance = graph_full[path[i-1]][name]['weight']
                    cumulative += distance
                    rows.append({
                        "Редослед": i+1,
                        "UIC": uic,
                        "Службено место": name,
                        "Растојање између сл. места (км)": f"{distance:.2f}",
                        "Кумулативно растојање (км)": f"{cumulative:.2f}"
                    })
            df_route = pd.DataFrame(rows)
            st.table(df_route)
        else:
            # Само чворне станице + почетна и крајња
            filtered = [p for p in path if p in junctions or p==path[0] or p==path[-1]]
            rows = []
            cumulative = 0.0
            for i, name in enumerate(path):
                if i < len(path)-1:
                    next_name = path[i+1]
                    distance = graph_full[name][next_name]['weight']
                else:
                    distance = 0.0
                if name in filtered:
                    uic_row = df[df['Назив службеног места']==name].iloc[0]
                    uic_value = uic_row['Шифра службеног места - UIC']
                    uic = "" if pd.isna(uic_value) else str(int(uic_value))
                    rows.append({
                        "Редослед": len(rows)+1,
                        "UIC": uic,
                        "Службено место": name,
                        "Кумулативно растојање (км)": f"{cumulative:.2f}"
                    })
                cumulative += distance
            df_short = pd.DataFrame(rows)
            st.table(df_short)

        if st.session_state["alt_paths"]:
            st.info(f"📌 {len(st.session_state['alt_paths'])} алтернативних рута.")

with col2:
    if st.session_state.get("show_map", False):
        if st.session_state.get("base_path"):
            coords=[]
            for name in st.session_state["base_path"]:
                row = df[df['Назив службеног места']==name].iloc[0]
                lat,lon = row['Координата1'], row['Координата2']
                if lat!=0 and lon!=0:
                    coords.append((lat,lon))
            if coords:
                m=folium.Map(location=coords[0], zoom_start=7, tiles='CartoDB positron')
                PolyLine(coords, color="red", weight=5, tooltip="Најкраћа рута").add_to(m)
                junctions_on_path = [n for n in st.session_state["base_path"] if n in junctions]
                for n in junctions_on_path:
                    row = df[df['Назив службеног места']==n].iloc[0]
                    lat,lon = row['Координата1'], row['Координата2']
                    if lat!=0 and lon!=0:
                        Marker((lat,lon), icon=folium.Icon(color="blue", icon="star"),
                               tooltip=f"Чворна: {n}").add_to(m)
                colors=["green","purple","orange","cadetblue","darkred"]
                for i,(p,d) in enumerate(st.session_state["alt_paths"]):
                    coords_alt=[]
                    for name in p:
                        row = df[df['Назив службеног места']==name].iloc[0]
                        lat,lon=row['Координата1'],row['Координата2']
                        if lat!=0 and lon!=0:
                            coords_alt.append((lat,lon))
                    if coords_alt:
                        PolyLine(coords_alt, color=colors[i%len(colors)], weight=3,
                                 tooltip=f"Алт. рута #{i+1} ({d:.1f} km)").add_to(m)
                st_folium(m, width=900, height=600)
            else:
                st.error("⚠️ Нема координата за приказ мапе.")
        else:
            st.info("ℹ️ Прво израчунај руту.")
    else:
        m=folium.Map(location=[44.0, 20.9], zoom_start=7, tiles='CartoDB positron')
        for _, row in df.iterrows():
            lat, lon = row['Координата1'], row['Координата2']
            if lat!=0 and lon!=0:
                folium.CircleMarker(location=(lat,lon), radius=2, color='gray', fill=True, fill_opacity=0.6).add_to(m)
        for name in junctions:
            row = df[df['Назив службеног места']==name].iloc[0]
            lat, lon = row['Координата1'], row['Координата2']
            if lat!=0 and lon!=0:
                Marker((lat,lon), icon=folium.Icon(color="blue", icon="star"),
                       tooltip=f"Чворна: {name}").add_to(m)
        st_folium(m, width=900, height=600)
