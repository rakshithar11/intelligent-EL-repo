import re
import pandas as pd
import plotly.express as px
import streamlit as st

from database import db
from ml import ingest

PALETTE = ["#E97843", "#F4B58E", "#A9BBAA", "#D7C5A6", "#B9A9D6",
           "#91B8D0", "#D8A5A5", "#F39A63", "#7FA08A", "#C9B28F"]


def _style(fig, height=340):
    fig.update_layout(
        height=height, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="DM Sans", color="#56616B", size=11),
        title_font=dict(family="Outfit", size=15, color="#303A46"),
        margin=dict(l=35, r=15, t=50, b=60), legend=dict(font=dict(size=10)))
    return fig


def _header(title, caption):
    st.markdown('<div class="section-label">RESEARCH & INSIGHTS</div>',
                unsafe_allow_html=True)
    st.title(title)
    st.caption(caption)
    st.divider()


def _split_names(value):
    parts = re.split(r"[;,&]|\band\b", value or "")
    return [p.strip() for p in parts if p.strip()]


def _mentor_map(projects):
    people = {}
    for p in projects:
        for name in _split_names(p.get("mentor")):
            key = re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
            key = re.sub(r"^(dr|prof|mr|ms|mrs)\s+", "", key)
            entry = people.setdefault(key, {"name": name, "projects": []})
            entry["projects"].append(p)
    return people


# =========================================================
# RESEARCH ANALYTICS
# =========================================================

def show_analytics(projects):

    _header("Research Analytics",
            "Trends across all EL projects in the repository.")

    if not projects:
        st.info("Upload projects to see analytics.")
        return

    raw = pd.DataFrame(projects)
    df = raw.copy()
    df["domain"] = df["domain"].fillna("").replace("", "Unclassified")
    df["year"] = df["year"].fillna("").astype(str).replace("", "Unknown")

    mentors = _mentor_map(projects)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Projects", len(df))
    c2.metric("Domains", df["domain"].nunique())
    c3.metric("Academic Years", (df["year"] != "Unknown").sum()
              and df.loc[df["year"] != "Unknown", "year"].nunique() or 0)
    c4.metric("Mentors", len(mentors))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(
        ["By domain", "By year", "Domain trends", "Contributors",
         "Gaps & data quality"])

    with t1:
        counts = df["domain"].value_counts().reset_index()
        counts.columns = ["Domain", "Projects"]
        a, b = st.columns(2)
        fig = px.bar(counts.sort_values("Projects"), x="Projects", y="Domain",
                     orientation="h", title="Projects per domain",
                     color_discrete_sequence=["#F39A63"])
        a.plotly_chart(_style(fig), use_container_width=True)
        fig2 = px.pie(counts, names="Domain", values="Projects", hole=0.6,
                      title="Domain share", color_discrete_sequence=PALETTE)
        b.plotly_chart(_style(fig2), use_container_width=True)

    with t2:
        years = df["year"].value_counts().reset_index()
        years.columns = ["Year", "Projects"]
        years = years.sort_values("Year")
        fig = px.bar(years, x="Year", y="Projects",
                     title="Projects per academic year",
                     color_discrete_sequence=["#E96D35"])
        st.plotly_chart(_style(fig), use_container_width=True)

    with t3:
        trend = df.groupby(["year", "domain"]).size().reset_index(name="Projects")
        trend = trend.sort_values("year")
        fig = px.bar(trend, x="year", y="Projects", color="domain",
                     title="Domains over the years",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(xaxis_title="Year", legend_title_text="Domain")
        st.plotly_chart(_style(fig, 400), use_container_width=True)

    with t4:
        if mentors:
            top = pd.DataFrame(
                [{"Mentor": m["name"], "Projects": len(m["projects"])}
                 for m in mentors.values()]
            ).sort_values("Projects", ascending=False).head(10)
            fig = px.bar(top.sort_values("Projects"), x="Projects", y="Mentor",
                         orientation="h", title="Top mentors by projects",
                         color_discrete_sequence=["#E97843"])
            st.plotly_chart(_style(fig), use_container_width=True)
        else:
            st.info("No mentor names recorded yet. Add them in Upload "
                    "Documents or by editing a project in Browse Projects.")

    with t5:
        st.subheader("Domain coverage")
        st.caption("Domains with few or no projects may be research "
                   "opportunities.")
        counts = df["domain"].value_counts()
        all_domains = sorted(set(ingest.DOMAINS) | set(counts.index))
        cov = pd.DataFrame({
            "Domain": all_domains,
            "Projects": [int(counts.get(d, 0)) for d in all_domains]
        }).sort_values("Projects")
        cov["Status"] = cov["Projects"].apply(
            lambda n: "No projects yet" if n == 0
            else ("Under-explored" if n <= 2 else "Well covered"))
        st.dataframe(cov, hide_index=True, use_container_width=True)

        st.subheader("Data completeness")
        st.caption("Share of projects with each optional field filled in.")

        def filled(col):
            s = raw[col].fillna("").astype(str).str.strip()
            return round(100 * (s != "").mean())

        comp = pd.DataFrame({
            "Field": ["Year", "Team", "Mentor"],
            "Filled (%)": [filled("year"), filled("team"), filled("mentor")]
        })
        fig = px.bar(comp, x="Field", y="Filled (%)", range_y=[0, 100],
                     color_discrete_sequence=["#A9BBAA"])
        st.plotly_chart(_style(fig, 300), use_container_width=True)


# =========================================================
# FACULTY DIRECTORY
# =========================================================

def show_faculty(projects):

    _header("Faculty Directory",
            "Find mentors and teams, and the projects they are connected to.")

    tab_m, tab_t = st.tabs(["Mentors", "Teams"])

    with tab_m:
        mentors = _mentor_map(projects)

        if not mentors:
            st.info("No mentor names recorded yet. Add them in Upload "
                    "Documents or by editing a project in Browse Projects.")
        else:
            q = st.text_input("Search mentor", placeholder="Type a name...")
            people = sorted(mentors.values(),
                            key=lambda m: -len(m["projects"]))
            if q:
                people = [m for m in people if q.lower() in m["name"].lower()]
            st.write(f"**{len(people)} mentor(s)**")

            for m in people:
                doms = sorted({p["domain"] for p in m["projects"]
                               if p.get("domain")})
                n = len(m["projects"])
                label = (f"{m['name']}  |  {n} project{'s' if n != 1 else ''}"
                         f"  |  {', '.join(doms)}")
                with st.expander(label):
                    for p in m["projects"]:
                        st.write(
                            f"• **{p['title']}**  ({p['domain']}, "
                            f"{p.get('year') or 'year n/a'}, "
                            f"team: {p.get('team') or '-'})")

    with tab_t:
        rows = {}
        for p in projects:
            team = (p.get("team") or "").strip()
            if not team:
                continue
            r = rows.setdefault(team.lower(), {
                "Team": team, "Mentor": set(), "Projects": 0,
                "Domains": set(), "Years": set()})
            r["Projects"] += 1
            r["Mentor"].update(_split_names(p.get("mentor")))
            if p.get("domain"):
                r["Domains"].add(p["domain"])
            if p.get("year"):
                r["Years"].add(str(p["year"]))

        if not rows:
            st.info("No team names recorded yet. Add them in Upload "
                    "Documents or by editing a project in Browse Projects.")
                else:
            table = pd.DataFrame([{
                "Team": r["Team"],
                "Mentor": ", ".join(sorted(r["Mentor"])) or "-",
                "Projects": r["Projects"],
                "Domains": ", ".join(sorted(r["Domains"])),
                "Years": ", ".join(sorted(r["Years"])) or "-",
            } for r in rows.values()]).sort_values("Team")
            st.write(f"**{len(table)} team(s)**")
            st.dataframe(table, hide_index=True, use_container_width=True)
