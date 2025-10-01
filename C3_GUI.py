import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from Optimize import photosynthesis_optimize
from C3 import Photosynthesis_C3_ACi
import io

st.set_page_config(layout="wide")

# -----------------------------
# Helper Functions
# -----------------------------

def get_inputs():
    """Collects climate and observation inputs from user via Streamlit UI."""
    # Default climate data
    data = {
        'Temperature': [25, "\u2070" + 'C'],
        'Pressure': [101.325, 'kPa'],
        'O2': [210, 'kPa']
    }
    df1 = pd.DataFrame(data).T
    df1.columns = ['value','units']
    df1.index.name = 'Parameter'

    # Default experimental data
    flags  = np.array([1,1,1,1,2,2,2,2,2,3,3])
    a = np.array([-3.3,3.5,10.8,17.8,23.5,27.6,30.5,31.7,33.0,33.3,33.3])
    ci_ppm = np.array([20.7,77.5,135,197,266,344,428,517,606,698,791])
    df2 = pd.DataFrame([flags, a, ci_ppm], index=['flag','Anet','Ci']).T

    # Layout
    col1, col2 = st.columns([1.45, 1.75])

    with col1:
        left,center,right = st.columns([0.3,1,0.3])
        with center:
            st.latex(r'''\text{Climate} ''')
            edited_df1 = st.data_editor(df1, num_rows='fixed',use_container_width=True, width=300, key="editor", height=150, )
        st.latex(r'''\text{Enter Param} ''')
        edited_df2 = st.data_editor(df2, num_rows="dynamic", use_container_width=True, key="editor2", height=500)

    # Extract user inputs
    ObsCi = pd.to_numeric(edited_df2['Ci'], errors='coerce').values
    ObsAnet = pd.to_numeric(edited_df2['Anet'], errors='coerce').values

    # Build input DataFrame
    Input = pd.DataFrame({
        'Ci': np.linspace(0, ObsCi[-1] + 200, 50),
        'temperature': edited_df1.iloc[0,0],
        'pressure': edited_df1.iloc[1,0],
        'O2': edited_df1.iloc[2,0]
    })

    return edited_df1, edited_df2, ObsCi, ObsAnet, Input, col2


def run_model(Input, edited_df2):
    """Runs photosynthesis optimization at Tleaf and 25°C."""
    # Initialize results DataFrame
    Photosynthesis = pd.DataFrame(
        data=np.zeros((5, 3)),
        index=['Vc_max','J','TPU','Rd*','gm*'],
        columns=['@T_leaf','@25','units']
    )
    Photosynthesis.loc[:,'units'] = 'μmol m⁻² s⁻¹'
    Photosynthesis.loc['gm*','units'] = 'μmol m⁻² s⁻¹ / Pa'

    # Optimization at Tleaf
    photo_vector, flag1, flag2, flag3 = photosynthesis_optimize(Input, edited_df2)
    Photosynthesis['@T_leaf'] = photo_vector

    # Optimization at 25°C
    Input.loc[0,'temperature'] = 25
    photo_vector, _, _, _ = photosynthesis_optimize(Input, edited_df2)
    Photosynthesis['@25'] = photo_vector

    # Constants
    Constants = {'R': 8.314/1000}  # kJ/mol·K

    # Run C3 photosynthesis model
    keys = ['vcmax25','jmax25','tpu25','rd25','gm25']
    Photosynthesis_dict = dict(zip(keys, photo_vector))
    model = Photosynthesis_C3_ACi(Input, Photosynthesis_dict)
    LeafMassFlux, LeafState = model.solve()

    return Photosynthesis, LeafMassFlux, (flag1, flag2, flag3)


def plot_results(col2, Input, LeafMassFlux, ObsCi, ObsAnet, Photosynthesis):
    """Plots A/Cc curve and displays outputs in UI."""
    with col2:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 2.5)

        x = Input['Ci']
        ax.plot(x, (LeafMassFlux['ac']-LeafMassFlux['rd']), c='r', linewidth=1, label='Rubisco')
        ax.plot(x, (LeafMassFlux['aj']-LeafMassFlux['rd']), c='b', linewidth=1, label='RuBP_regen')
        ax.plot(x, (LeafMassFlux['ap']-LeafMassFlux['rd']), c='y', linewidth=1, label='TPU')
        ax.scatter(ObsCi, ObsAnet, s=20, marker='o', label='Aobs')

        ax.set_xlabel('Ci (Pa)')
        ax.set_ylabel('A (μmol m⁻² s⁻¹)')
        ax.set_title('A/Ci Curve')
        ax.legend()
        left, center, right = st.columns([0.2,1,0.5])
        with center:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=500, bbox_inches="tight")
            st.image(buf, caption="A/Ci", width=800)
        left, center, right = st.columns([0.45,1.2,0.3])
        with center:
            st.latex(r'''\text{Outputs} ''')
            st.dataframe(Photosynthesis, use_container_width=True, height=225)


# -----------------------------
# Main App
# -----------------------------
def main():
    edited_df1, edited_df2, ObsCi, ObsAnet, Input, col2 = get_inputs()
    c1, c2 = st.columns([1.45, 1.75])
    with c1:
        l,c,r = st.columns([1,1,1])
        with c:
            if st.button('Calculate'):
                Photosynthesis, LeafMassFlux, flags = run_model(Input, edited_df2)
                plot_results(col2, Input, LeafMassFlux, ObsCi, ObsAnet, Photosynthesis)

        # Display diagnostic flags
        #   st.write(f'flags 1: {flags[0]}  \nflags 2: {flags[1]}  \nflags 3: {flags[2]}')


if __name__ == "__main__":
    main()
