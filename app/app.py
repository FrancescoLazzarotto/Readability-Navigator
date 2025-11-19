import streamlit as st
from main import main
import pandas as pd
import numpy as np
""


st.title("Readability Navigator")


df = main()
st.write(df.head())
st.button("Genera Raccomandazioni") 
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)



