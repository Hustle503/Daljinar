import streamlit as st
import pandas as pd
import networkx as nx
import folium
from folium import PolyLine, Marker
from streamlit_folium import st_folium

st.set_page_config(page_title="Најкраћа рута + мапа", layout="wide")
st.title("🛤️ Израчунавање најкраће руте")

@st.cache_data
def load_data():
    return pd.read_excel("Izjava o mreži 2025.xlsx")

df = load_data()
junctions = df[df['Назив службеног места'].duplicated(keep=False)]['Назив службеног места'].unique()
rasputnice = ["РАСПУТНИЦА А", "РАСПУТНИЦА Б", "РАСПУТНИЦА Г", "РАСПУТНИЦА К", "РАСПУТНИЦА К1", "РАСПУТНИЦА Р", "РАСПУТНИЦА Т"]
key_juncs = ["НИШ", "ВЕЛИКА ПЛАНА", "МАЛА КРСНА", "РЕСНИК", "БЕОГРАД ЦЕНТАР", "НОВИ САД", "ПАНЧЕВО ГЛАВНА", "ОРЛОВАТ"]

# Градимо граф
grouped = df.groupby('Број пруге')
edges = []
for _, grp in grouped:
    grp = grp.sort_values('Редни број службеног места на прузи')
    prev = None
    for _, r in grp.iterrows():
        curr = r['Назив службеног места']
        if prev:
            dist = r['Километарска удаљеност'] / 1000
            edges.append((prev, curr, dist))
        prev = curr
G = nx.Graph()
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

def shortest_multi(G, start, vias, end):
    path = []
    used = set()
    cur = start
    Gc = G.copy()
    for via in vias + [end]:
        segment = nx.shortest_path(Gc, cur, via, weight='weight')
        for node in segment:
            if node != via:
                used.add(node)
        Gc.remove_nodes_from([n for n in used if n != via])
        path += segment if not path else segment[1:]
        cur = via
    dist = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
    return path, dist

# Проналажење алтернативних рута (са обилазницама ако прва иде кроз БГ ЦЕНТАР)
def find_alternatives(G, base, n, allow_return=False):
    alts = []
    base_nodes = set(base)
    first_alt = None
    include_bg_centar = any(r in base_nodes for r in rasputnice)

    if include_bg_centar:
        try:
            path1, dist1 = shortest_multi(G, base[0], ["БЕОГРАД ЦЕНТАР"], base[-1])
            if path1 != base:
                alts.append((path1, dist1))
                first_alt = path1
        except:
            pass

    reference = first_alt if first_alt else base

    # Обилазница преко Панчева и Орловата ако имамо БГ Центар и Нови Сад
    if include_bg_centar and any("НОВИ САД" in s for s in reference):
        try:
            path_ob, dist_ob = shortest_multi(G, base[0], ["ПАНЧЕВО ГЛАВНА", "ОРЛОВАТ"], base[-1])
            if path_ob not in [a[0] for a in alts] and path_ob != base:
                alts.append((path_ob, dist_ob))
        except:
            pass

    if any("НОВИ САД" in s for s in reference):
        G2 = G.copy()
        for u in list(G2.nodes):
            if "НОВИ САД" in u:
                G2.remove_node(u)
        try:
            p_ns, d_ns = shortest_multi(G2, base[0], [], base[-1])
            if p_ns not in [a[0] for a in alts]:
                alts.append((p_ns, d_ns))
        except:
            pass

    if any("СУБОТИЦА" in s for s in reference):
        G3 = G.copy()
        for u in list(G3.nodes):
            if "СУБОТИЦА" in u:
                G3.remove_node(u)
        try:
            p_sb, d_sb = shortest_multi(G3, base[0], [], base[-1])
            if p_sb not in [a[0] for a in alts]:
                alts.append((p_sb, d_sb))
        except:
            pass

    for kj in key_juncs:
        if kj in reference and len(alts) >= n:
            break
        if kj in reference:
            G2 = G.copy()
            G2.remove_node(kj)
            try:
                p2, d2 = shortest_multi(G2, base[0], [], base[-1])
                if p2 not in [base] + [a[0] for a in alts]:
                    alts.append((p2, d2))
            except:
                pass

    if not allow_return:
        filtered = []
        for p, d in alts:
            edges = set((min(u, v), max(u, v)) for u, v in zip(p[:-1], p[1:]))
            if len(edges) == len(p) - 1:
                filtered.append((p, d))
        alts = filtered

    if len(alts) < n:
        st.warning(f"⚠️ Тражено: {n}, али пронађено само {len(alts)} валидних алтернативних рута.")

    return alts[:n]

# --- UI ---
stations = sorted(df['Назив службеног места'].unique())
def_start = stations.index("ДРЖАВНА ГРАНИЦА ПРЕШЕВО") if "ДРЖАВНА ГРАНИЦА ПРЕШЕВО" in stations else 0
def_end = stations.index("ДРЖАВНА ГРАНИЦА СУБОТИЦА") if "ДРЖАВНА ГРАНИЦА СУБОТИЦА" in stations else -1

col1, col2 = st.columns([1.2, 2])
with col1:
    start = st.selectbox("🚩 Почетна станица", stations, index=def_start)
    end = st.selectbox("🏁 Крајња станица", stations, index=def_end)
    vias = st.multiselect("🔀 Међустанице", stations)
    exclude = st.multiselect("🚫 Искључи службена места", stations)
    allow_return = st.checkbox("🔄 Дозволи повратне руте", value=False)
    n_alt = st.selectbox("➕ Алтернативне руте", [0, 1, 2, 3, 4, 5], index=0)
    show_all = st.checkbox("👀 Прикажи све станице", value=False)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Израчунај руту"):
            GG = G.copy()
            if exclude:
                GG.remove_nodes_from(exclude)
            base, bd = shortest_multi(GG, start, vias, end)
            st.session_state['base'] = base
            st.session_state['bd'] = bd
            st.session_state['alts'] = find_alternatives(GG, base, n_alt, allow_return)
            st.session_state['show_map'] = False
    with c2:
        if st.button("Прикажи мапу"):
            st.session_state['show_map'] = True

    if st.session_state.get('base'):
        st.success(f"✅ Најкраћа рута пронађена! {st.session_state['bd']:.2f} km")

        def build_table(path):
            rows, cum = [], 0.0
            for i, n in enumerate(path):
                row = df[df['Назив службеног места'] == n]
                uic = "" if row['Шифра службеног места - UIC'].isna().any() else str(int(row['Шифра службеног места - UIC'].iloc[0]))
                dist = 0.0 if i == 0 else G[path[i - 1]][n]['weight']
                cum += dist
                rows.append({
                    "Ред": i + 1,
                    "UIC": uic,
                    "Службено место": n,
                    "Растојање између сл. места": f"{dist:.2f}",
                    "Кумулативно растојање (км)": f"{cum:.2f}"
                })
            dfp = pd.DataFrame(rows)
            if not show_all:
                dfp = dfp[dfp['Службено место'].isin(junctions) | (dfp['Ред'] == 1) | (dfp['Службено место'] == path[-1])]
                dfp = dfp.drop(columns=["Растојање између сл. места"])
            return dfp

        st.table(build_table(st.session_state['base']))
        for i, (p, d) in enumerate(st.session_state['alts']):
            st.markdown(f"### 🛤️ Алт. рута #{i+1}: {d:.2f} km")
            st.table(build_table(p))

with col2:
    if st.session_state.get('show_map') and 'base' in st.session_state:
        m = folium.Map(tiles="CartoDB positron")
        all_coords = []

        for p, d in [(st.session_state['base'], None)] + st.session_state['alts']:
            coords = [
                tuple(df.loc[df['Назив службеног места'] == name][['Координата1', 'Координата2']].iloc[0])
                for name in p if tuple(df.loc[df['Назив службеног места'] == name][['Координата1', 'Координата2']].iloc[0]) != (0, 0)
            ]
            all_coords += coords
            color = 'red' if d is None else 'blue'
            PolyLine(coords, color=color, weight=5 if d is None else 3).add_to(m)

        if all_coords:
            m.fit_bounds(all_coords)

        def mark(name, color, icon):
            row = df[df['Назив службеног места'] == name].iloc[0]
            Marker([row['Координата1'], row['Координата2']],
                   icon=folium.Icon(color=color, icon=icon),
                   tooltip=name).add_to(m)

        base = st.session_state['base']
        mark(base[0], 'green', 'flag')
        mark(base[-1], 'red', 'flag')
        for v in vias:
            mark(v, 'blue', 'flag')
        for j in base:
            if j in junctions:
                mark(j, 'blue', 'star')

        st_folium(m, width=700, height=600)

    else:
        m = folium.Map(location=[44.5, 20.9], zoom_start=7, tiles="CartoDB positron")

        for _, group in grouped:
            group_sorted = group.sort_values("Редни број службеног места на прузи")
            coords = []
            for _, row in group_sorted.iterrows():
                lat, lon = row["Координата1"], row["Координата2"]
                if lat != 0 and lon != 0:
                    coords.append((lat, lon))
            if len(coords) >= 2:
                folium.PolyLine(coords, color="black", weight=2).add_to(m)

        for _, row in df.iterrows():
            name = row["Назив службеног места"]
            lat, lon = row["Координата1"], row["Координата2"]
            if lat == 0 or lon == 0:
                continue
            if name in junctions or name.startswith("ДРЖАВНА ГРАНИЦА"):
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=2,
                    color="black",
                    fill=True,
                    fill_opacity=0.8
                ).add_to(m)

        st_folium(m, width=700, height=600)
