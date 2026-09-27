import math
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

st.set_page_config(page_title="MechCalc", page_icon="⚙️", layout="wide")

# -------------------------
# Engineering data/constants
# -------------------------
MATERIALS = {
    "Structural Steel": {"rho": 7850, "yield": 250e6, "E": 200e9, "G": 80e9},
    "AISI 4130 Steel": {"rho": 7850, "yield": 435e6, "E": 205e9, "G": 80e9},
    "AISI 4340 Steel": {"rho": 7850, "yield": 470e6, "E": 200e9, "G": 80e9},
    "Aluminium 6061-T6": {"rho": 2700, "yield": 276e6, "E": 69e9, "G": 26e9},
    "Aluminium 7075-T6": {"rho": 2810, "yield": 503e6, "E": 71.7e9, "G": 26.9e9},
}
G0 = 9.81

# -------------------------
# Calculation functions
# -------------------------
def beam(L,b,h,F,E,yield_strength):
    I=b*h**3/12; M=F*L/4; stress=M*(h/2)/I; delta=F*L**3/(48*E*I)
    return {"Moment_Nm":M,"SecondMoment_m4":I,"Stress_MPa":stress/1e6,"Deflection_mm":delta*1000,"FoS":yield_strength/stress}

def shaft(L,d,T,G,yield_strength,rho):
    J=math.pi*d**4/32; tau=T*(d/2)/J; twist=math.degrees(T*L/(G*J)); mass=rho*math.pi*d**2/4*L
    return {"PolarMoment_m4":J,"Mass_kg":mass,"ShearStress_MPa":tau/1e6,"Twist_deg":twist,"FoS":0.577*yield_strength/tau}

def axial(F,A,E,L):
    stress=F/A; eps=stress/E
    return {"Stress_MPa":stress/1e6,"Strain":eps,"Elongation_mm":eps*L*1000}

def spring_force(k,x): return {"Force_N":k*x,"Energy_J":0.5*k*x*x}
def force_components(F,theta): return {"Fx_N":F*math.cos(math.radians(theta)),"Fy_N":F*math.sin(math.radians(theta))}
def euler_column(E,I,K,L): return math.pi**2*E*I/(K*L)**2
def beam_reactions(P,a,L): return {"RA_N":P*(L-a)/L,"RB_N":P*a/L}
def bending_stress(M,c,I): return M*c/I
def power(T,rpm): return T*2*math.pi*rpm/60
def torque(P,rpm): return P/(2*math.pi*rpm/60)
def gear(z1,z2,n1,T1,eta): return {"GearRatio":z2/z1,"OutputRPM":n1*z1/z2,"OutputTorque_Nm":T1*z2/z1*eta}
def belt_speed(D,rpm): return math.pi*D*rpm/60
def belt_tension(P,v): return P/v
def spring_rate(G,d,D,n): return G*d**4/(8*D**3*n)
def shaft_diameter_torsion(T,tau): return (16*T/(math.pi*tau))**(1/3)
def flywheel_energy(I,w1,w2): return 0.5*I*(w1*w1-w2*w2)
def flywheel_inertia(E,w1,w2): return 2*E/(w1*w1-w2*w2)
def natural_frequency(k,m): return math.sqrt(k/m)/(2*math.pi)
def damped_frequency(fn,zeta): return fn*math.sqrt(max(0,1-zeta*zeta))
def damping_ratio(c,k,m): return c/(2*math.sqrt(k*m))
def critical_damping(k,m): return 2*math.sqrt(k*m)
def reynolds(rho,v,D,mu): return rho*v*D/mu
def velocity_from_flow(Q,D): return 4*Q/(math.pi*D**2)
def pressure_from_head(rho,g,h): return rho*g*h
def darcy_drop(f,L,D,rho,v): return f*L/D*rho*v*v/2
def bernoulli_velocity(dp,rho): return math.sqrt(max(0,2*dp/rho))
def ideal_pressure(m,R,T,V): return m*R*T/V
def ideal_temperature(P,R,m,V): return P*V/(m*R)
def delta_u(m,cv,T1,T2): return m*cv*(T2-T1)
def delta_h(m,cp,T1,T2): return m*cp*(T2-T1)
def entropy_change(m,cp,T1,T2): return m*cp*math.log(T2/T1)
def isentropic_T(T1,p1,p2,gamma): return T1*(p2/p1)**((gamma-1)/gamma)
def efficiency(out,inp): return out/inp
def conduction(k,A,dt,L): return k*A*dt/L
def convection(h,A,dt): return h*A*dt
def thermal_expansion(alpha,L,dt): return alpha*L*dt
def radiation(eps,A,T1,T2): return eps*5.670374419e-8*A*(T1**4-T2**4)
def thermal_resistance(L,k,A): return L/(k*A)
def circle_area(D): return math.pi*D**2/4
def cylinder_volume(D,L): return math.pi*D**2*L/4
def cone_volume(D,h): return math.pi*D**2*h/12
def sphere_volume(D): return math.pi*D**3/6
def interpolation(x1,x2,y1,y2,x): return y1+(x-x1)*(y2-y1)/(x2-x1)
def cutting_rpm(Vc,D): return 1000*Vc/(math.pi*D)
def milling_feed(fz,z,N): return fz*z*N
def machining_time(L,f,N): return L/(f*N)
def turning_mrr(D,d,f,N): return math.pi*D*d*f*N
def milling_mrr(w,ap,fz,z,N): return w*ap*fz*z*N
def casting_shrinkage(L_pattern,L_cast): return (L_pattern-L_cast)/L_cast*100
def robot_force(m,a,Crr,r,grade,g=G0):
    F=m*a+Crr*m*g*math.cos(math.radians(grade))+m*g*math.sin(math.radians(grade)); return F,F*r
def wheel_rpm(v,D): return v*60/(math.pi*D)
def suspension_load(mass,weight_dist,ax,ay): return mass*G0*weight_dist + mass*ax*0.0 + mass*ay*0.0
def suspension_frequency(k,m): return math.sqrt(k/m)/(2*math.pi)
def chassis_bending_stress(M,c,I): return M*c/I
def chassis_torsion_stress(T,r,J): return T*r/J
def brake_torque(mu,P,r_eff): return mu*P*r_eff
def brake_disc_energy(m_vehicle,v1,v2): return 0.5*m_vehicle*(v1*v1-v2*v2)
def brake_disc_temperature(E,m_disc,cp): return E/(m_disc*cp)
def ackermann_outer_angle(L,R): return math.degrees(math.atan(L/(R+0)))
def unit_convert(category,value,unit):
    maps={
        "Length":{"m":1,"mm":1e-3,"cm":1e-2,"inch":0.0254,"ft":0.3048},
        "Force":{"N":1,"kN":1000,"lbf":4.4482216152605},
        "Pressure":{"Pa":1,"kPa":1000,"MPa":1e6,"bar":1e5,"psi":6894.757293168},
        "Power":{"W":1,"kW":1000,"hp":745.6998715822702},
        "Torque":{"N·m":1,"N·mm":.001,"lbf·ft":1.3558179483314},
    }
    if category=="Temperature":
        C=value if unit=="°C" else value-273.15 if unit=="K" else (value-32)*5/9
        return {"°C":C,"K":C+273.15,"°F":C*9/5+32}
    base=value*maps[category][unit]; return {u:base/f for u,f in maps[category].items()}

# -------------------------
# Equation solvers
# -------------------------
def solve_linear2(a1,b1,c1,a2,b2,c2):
    det=a1*b2-a2*b1
    if abs(det)<1e-12: raise ValueError("The two equations do not have a unique solution.")
    return (c1*b2-c2*b1)/det,(a1*c2-a2*c1)/det

def polynomial_roots(coeffs):
    coeffs=np.array(coeffs,dtype=float)
    nz=np.flatnonzero(np.abs(coeffs)>1e-14)
    if len(nz)==0: raise ValueError("This is the zero polynomial.")
    coeffs=coeffs[nz[0]:]
    roots=np.roots(coeffs)
    return roots

# -------------------------
# History / Excel
# -------------------------
if "history" not in st.session_state: st.session_state.history=[]
if "last_history_signature" not in st.session_state: st.session_state.last_history_signature=None

def add_history(module,tool,inputs,outputs):
    clean_inputs={k:v for k,v in inputs.items() if v is not None}
    clean_outputs={k:v for k,v in outputs.items() if v is not None}
    sig=repr((module,tool,clean_inputs,clean_outputs))
    if sig==st.session_state.last_history_signature: return
    st.session_state.history.append({"Module":module,"Tool":tool,"Inputs":clean_inputs,"Outputs":clean_outputs})
    st.session_state.last_history_signature=sig

def get_active_tool(module):
    return st.session_state.get("active_tool")

def history_dataframe(module,tool):
    rows=[]
    for i,item in enumerate([x for x in st.session_state.history if x["Module"]==module and x["Tool"]==tool],1):
        row={"Iteration":i}; row.update(item["Inputs"]); row.update(item["Outputs"]); rows.append(row)
    return pd.DataFrame(rows)

def excel_for_history(module,tool):
    df=history_dataframe(module,tool); out=BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as writer:
        sheet=(tool or module)[:31].replace("/","-").replace("\\","-"); df.to_excel(writer,index=False,sheet_name=sheet)
        ws=writer.book[sheet]; ws.freeze_panes="A2"
        for cell in ws[1]:
            cell.font=Font(bold=True,color="FFFFFF"); cell.fill=PatternFill("solid",fgColor="16324F"); cell.alignment=Alignment(horizontal="center")
        for col in ws.columns:
            width=min(max(len(str(c.value or "")) for c in col)+2,32); ws.column_dimensions[get_column_letter(col[0].column)].width=width
    out.seek(0); return out.getvalue()

def result_card(outputs):
    visible={k:v for k,v in outputs.items() if v is not None}; cols=st.columns(min(5,max(1,len(visible))))
    for c,(k,v) in zip(cols,visible.items()): c.metric(k,f"{v:.5g}" if isinstance(v,(float,int,np.floating,np.integer)) else str(v))

def formula_box(equation, note=None):
    st.markdown("**Governing formula**")
    st.latex(equation)
    if note: st.caption(note)

def finish(module,tool,inputs,outputs,equation):
    formula_box(equation)
    result_card(outputs)
    add_history(module,tool,inputs,outputs)

# -------------------------
# UI
# -------------------------
st.markdown("""<style>.block-container{padding-top:1.2rem;max-width:1500px}[data-testid='stMetric']{border:1px solid rgba(255,255,255,.10);border-radius:12px;padding:10px}.formula{padding:.35rem 0}</style>""",unsafe_allow_html=True)
st.title("⚙️ MechCalc")
st.caption("College Engineering Calculation Suite • enter real inputs → see the governing equation → calculate → repeat → export")

modules=["Home","Mechanics & Structures","Dynamics","Machine Design","Fluids","Thermodynamics","Heat Transfer","Engineering Math & Geometry","Manufacturing & Production","Robotics & Vehicle","Unit Converter"]
module=st.sidebar.selectbox("Engineering Toolbox",modules)

if module=="Home":
    st.subheader("Everyday engineering calculations in one place")
    st.write("A student-friendly calculation workspace for mechanical, production, civil-structure, thermal, fluid, robotics and vehicle work.")
    st.info("Results are engineering calculations for education and preliminary design. Always verify units, assumptions, material properties, standards, boundary conditions and safety factors before using a result in real hardware.")
    cols=st.columns(4)
    for c,t,d in [(cols[0],"🧱 Mechanics","Stress, beams, columns, reactions, shafts"),(cols[1],"⚙️ Machine Design","Gears, belts, springs, shafts, flywheels"),(cols[2],"🔥 Thermal & Fluids","Thermodynamics, heat transfer, pipe flow"),(cols[3],"🤖 Projects","Suspension, chassis, brakes, manufacturing")]:
        with c: st.markdown(f"### {t}"); st.write(d)
    st.divider(); st.subheader("Simple workflow"); st.code("Select module → select calculation → enter engineering data → read formula → calculate → repeat → export Excel")

elif module=="Mechanics & Structures":
    tools=["Beam Analysis","Simply Supported Beam Reactions","Solid Shaft Torsion","Axial Stress & Strain","Bending Stress","Euler Column Buckling","Force Components","Spring Force & Energy"]
    tool=st.selectbox("Calculation",tools); st.session_state.active_tool=tool
    if tool=="Beam Analysis":
        c1,c2=st.columns(2)
        with c1: L=st.number_input("Length L (m)",value=2.0,min_value=.001); b=st.number_input("Width b (mm)",value=50.,min_value=.1); h=st.number_input("Height h (mm)",value=100.,min_value=.1)
        with c2: F=st.number_input("Central load P (N)",value=1000.,min_value=0.); mat=st.selectbox("Material",list(MATERIALS))
        m=MATERIALS[mat]; r=beam(L,b/1000,h/1000,F,m["E"],m["yield"])
        finish(module,tool,{"Length_m":L,"Width_mm":b,"Height_mm":h,"Load_N":F,"Material":mat},r,r"I=\frac{bh^3}{12}\qquad M=\frac{PL}{4}\qquad \sigma=\frac{Mc}{I}\qquad \delta=\frac{PL^3}{48EI}")
        x=np.linspace(0,L,100); I=b/1000*(h/1000)**3/12; y=F*x*(3*L**2-4*x**2)/(48*m["E"]*I); st.plotly_chart(go.Figure(go.Scatter(x=x,y=y*1000,mode="lines")).update_layout(title="Deflection shape",xaxis_title="Position (m)",yaxis_title="Deflection (mm)"),use_container_width=True)
    elif tool=="Simply Supported Beam Reactions":
        L=st.number_input("Span L (m)",value=2.,min_value=.001); P=st.number_input("Point load P (N)",value=1000.,min_value=0.); a=st.number_input("Load distance from left support a (m)",value=1.,min_value=0.)
        a=min(a,L); r=beam_reactions(P,a,L); finish(module,tool,{"Span_m":L,"Load_N":P,"LoadPosition_m":a},r,r"R_A=\frac{P(L-a)}{L}\qquad R_B=\frac{Pa}{L}")
    elif tool=="Solid Shaft Torsion":
        L=st.number_input("Length L (m)",value=.5,min_value=.001); d=st.number_input("Diameter d (mm)",value=30.,min_value=.1); T=st.number_input("Torque T (N·m)",value=500.,min_value=0.); mat=st.selectbox("Material",list(MATERIALS)); m=MATERIALS[mat]; r=shaft(L,d/1000,T,m["G"],m["yield"],m["rho"]); finish(module,tool,{"Length_m":L,"Diameter_mm":d,"Torque_Nm":T,"Material":mat},r,r"J=\frac{\pi d^4}{32}\qquad \tau=\frac{Tr}{J}\qquad \theta=\frac{TL}{GJ}")
    elif tool=="Axial Stress & Strain":
        F=st.number_input("Axial force F (N)",value=10000.,min_value=0.); A=st.number_input("Area A (mm²)",value=100.,min_value=.001); E=st.number_input("Young's modulus E (GPa)",value=200.,min_value=.001); L=st.number_input("Original length L (m)",value=1.,min_value=.001); r=axial(F,A*1e-6,E*1e9,L); finish(module,tool,{"Force_N":F,"Area_mm2":A,"E_GPa":E,"Length_m":L},r,r"\sigma=\frac{F}{A}\qquad \epsilon=\frac{\sigma}{E}\qquad \delta=\epsilon L")
    elif tool=="Bending Stress":
        M=st.number_input("Bending moment M (N·m)",value=1000.,min_value=0.); c=st.number_input("Extreme-fibre distance c (mm)",value=25.,min_value=.001); I=st.number_input("Second moment I (mm⁴)",value=4.17e6,min_value=.001); s=bending_stress(M,c/1000,I*1e-12); finish(module,tool,{"Moment_Nm":M,"c_mm":c,"I_mm4":I},{"BendingStress_MPa":s/1e6},r"\sigma=\frac{Mc}{I}")
    elif tool=="Euler Column Buckling":
        E=st.number_input("E (GPa)",value=200.,min_value=.001)*1e9; I=st.number_input("I (mm⁴)",value=8e6,min_value=.001)*1e-12; L=st.number_input("Effective length L (m)",value=2.,min_value=.001); K=st.number_input("Effective length factor K",value=1.,min_value=.01); r={"CriticalLoad_N":euler_column(E,I,K,L)}; finish(module,tool,{"E_GPa":E/1e9,"I_mm4":I/1e-12,"Length_m":L,"K":K},r,r"P_{cr}=\frac{\pi^2EI}{(KL)^2}")
    elif tool=="Force Components":
        F=st.number_input("Force F (N)",value=1000.,min_value=0.); a=st.number_input("Angle θ from +X (deg)",value=30.); r=force_components(F,a); finish(module,tool,{"Force_N":F,"Angle_deg":a},r,r"F_x=F\cos\theta\qquad F_y=F\sin\theta")
    else:
        k=st.number_input("Spring stiffness k (N/m)",value=1000.,min_value=0.); x=st.number_input("Deflection x (m)",value=.02,min_value=0.); r=spring_force(k,x); finish(module,tool,{"k_N_per_m":k,"x_m":x},r,r"F=kx\qquad U=\frac12kx^2")

elif module=="Dynamics":
    tools=["Kinetic / Potential Energy","Momentum & Impulse","Natural Frequency","Damped Natural Frequency","Damping Ratio / Critical Damping","Flywheel Energy","Rotational Kinetic Energy","Projectile Range"]
    tool=st.selectbox("Calculation",tools); st.session_state.active_tool=tool
    if tool=="Kinetic / Potential Energy":
        m=st.number_input("Mass (kg)",value=10.,min_value=0.); v=st.number_input("Velocity (m/s)",value=5.,min_value=0.); h=st.number_input("Height (m)",value=2.,min_value=0.); r={"KineticEnergy_J":.5*m*v*v,"PotentialEnergy_J":m*G0*h}; finish(module,tool,{"Mass_kg":m,"Velocity_ms":v,"Height_m":h},r,r"KE=\frac12mv^2\qquad PE=mgh")
    elif tool=="Momentum & Impulse":
        m=st.number_input("Mass (kg)",value=10.,min_value=0.); v1=st.number_input("Initial velocity (m/s)",value=0.); v2=st.number_input("Final velocity (m/s)",value=5.); dt=st.number_input("Time interval (s)",value=.2,min_value=.0001); r={"MomentumChange_Ns":m*(v2-v1),"AverageForce_N":m*(v2-v1)/dt}; finish(module,tool,{"Mass_kg":m,"v1_ms":v1,"v2_ms":v2,"dt_s":dt},r,r"\Delta p=m(v_2-v_1)\qquad F_{avg}=\frac{\Delta p}{\Delta t}")
    elif tool=="Natural Frequency":
        k=st.number_input("Stiffness k (N/m)",value=10000.,min_value=.001); m=st.number_input("Mass m (kg)",value=10.,min_value=.001); fn=natural_frequency(k,m); finish(module,tool,{"k_N_per_m":k,"m_kg":m},{"NaturalFrequency_Hz":fn},r"f_n=\frac{1}{2\pi}\sqrt{\frac{k}{m}}")
    elif tool=="Damped Natural Frequency":
        k=st.number_input("Stiffness k (N/m)",value=10000.,min_value=.001); m=st.number_input("Mass m (kg)",value=10.,min_value=.001); z=st.number_input("Damping ratio ζ",value=.05,min_value=0.,max_value=.999); fn=natural_frequency(k,m); finish(module,tool,{"k_N_per_m":k,"m_kg":m,"zeta":z},{"Undamped_Hz":fn,"Damped_Hz":damped_frequency(fn,z)},r"f_d=f_n\sqrt{1-\zeta^2}")
    elif tool=="Damping Ratio / Critical Damping":
        k=st.number_input("Stiffness k (N/m)",value=10000.,min_value=.001); m=st.number_input("Mass m (kg)",value=10.,min_value=.001); c=st.number_input("Damping coefficient c (N·s/m)",value=50.,min_value=0.); z=c/(2*math.sqrt(k*m)); cc=critical_damping(k,m); finish(module,tool,{"k_N_per_m":k,"m_kg":m,"c_Ns_per_m":c},{"DampingRatio":z,"CriticalDamping_Ns_per_m":cc},r"\zeta=\frac{c}{2\sqrt{km}}\qquad c_c=2\sqrt{km}")
    elif tool=="Flywheel Energy":
        I=st.number_input("Mass moment I (kg·m²)",value=.5,min_value=.0001); n1=st.number_input("Initial RPM",value=1000.,min_value=0.); n2=st.number_input("Final RPM",value=800.,min_value=0.); w1=2*math.pi*n1/60; w2=2*math.pi*n2/60; E=flywheel_energy(I,w1,w2); finish(module,tool,{"I_kgm2":I,"InitialRPM":n1,"FinalRPM":n2},{"EnergyReleased_J":E},r"E=\frac12I(\omega_1^2-\omega_2^2)\qquad \omega=\frac{2\pi N}{60}")
    elif tool=="Rotational Kinetic Energy":
        I=st.number_input("Mass moment I (kg·m²)",value=.05,min_value=.0001); rpm=st.number_input("RPM",value=1000.,min_value=0.); w=2*math.pi*rpm/60; finish(module,tool,{"I_kgm2":I,"RPM":rpm},{"AngularSpeed_rad_s":w,"RotationalKE_J":.5*I*w*w},r"\omega=\frac{2\pi N}{60}\qquad KE=\frac12I\omega^2")
    else:
        v=st.number_input("Launch speed (m/s)",value=20.,min_value=0.); a=st.number_input("Launch angle (deg)",value=45.); R=v*v*math.sin(math.radians(2*a))/G0; finish(module,tool,{"Velocity_ms":v,"Angle_deg":a},{"Range_m":R},r"R=\frac{v^2\sin(2\theta)}{g}")

elif module=="Machine Design":
    tools=["Power ↔ Torque ↔ RPM","Gear Ratio & Output","Belt Speed","Belt Tension from Power","Spring Rate","Shaft Diameter from Torsion","Flywheel Inertia","Shaft Power from Torque"]
    tool=st.selectbox("Calculation",tools); st.session_state.active_tool=tool
    if tool=="Power ↔ Torque ↔ RPM":
        mode=st.radio("Calculate",["Power","Torque"],horizontal=True); rpm=st.number_input("RPM",value=1500.,min_value=.1)
        if mode=="Power": T=st.number_input("Torque (N·m)",value=100.,min_value=0.); p=power(T,rpm); finish(module,tool,{"Torque_Nm":T,"RPM":rpm},{"Power_W":p,"Power_kW":p/1000},r"P=T\omega\qquad \omega=\frac{2\pi N}{60}")
        else: P=st.number_input("Power (W)",value=5000.,min_value=0.); t=torque(P,rpm); finish(module,tool,{"Power_W":P,"RPM":rpm},{"Torque_Nm":t},r"T=\frac{P}{\omega}\qquad \omega=\frac{2\pi N}{60}")
    elif tool=="Gear Ratio & Output":
        z1=st.number_input("Driver teeth z₁",value=20,min_value=1); z2=st.number_input("Driven teeth z₂",value=60,min_value=1); n1=st.number_input("Input RPM",value=1500.,min_value=0.); T1=st.number_input("Input torque (N·m)",value=10.,min_value=0.); eta=st.number_input("Efficiency",value=.95,min_value=.01,max_value=1.); r=gear(z1,z2,n1,T1,eta); finish(module,tool,{"z1":z1,"z2":z2,"InputRPM":n1,"InputTorque_Nm":T1,"Efficiency":eta},r,r"i=\frac{z_2}{z_1}\qquad N_2=N_1\frac{z_1}{z_2}\qquad T_2=T_1\frac{z_2}{z_1}\eta")
    elif tool=="Belt Speed":
        D=st.number_input("Pulley diameter D (m)",value=.2,min_value=.0001); rpm=st.number_input("Pulley speed N (RPM)",value=1000.,min_value=0.); v=belt_speed(D,rpm); finish(module,tool,{"PulleyDiameter_m":D,"RPM":rpm},{"BeltSpeed_ms":v},r"v=\frac{\pi DN}{60}")
    elif tool=="Belt Tension from Power":
        P=st.number_input("Transmitted power P (W)",value=5000.,min_value=0.); v=st.number_input("Belt speed v (m/s)",value=10.,min_value=.001); T=belt_tension(P,v); finish(module,tool,{"Power_W":P,"BeltSpeed_ms":v},{"EffectiveTension_N":T},r"P=(T_1-T_2)v\qquad T_1-T_2=\frac{P}{v}")
    elif tool=="Spring Rate":
        G=st.number_input("Shear modulus G (Pa)",value=80e9,min_value=1.); d=st.number_input("Wire diameter d (m)",value=.003,min_value=.00001); D=st.number_input("Mean coil diameter D (m)",value=.03,min_value=.00001); n=st.number_input("Active coils n",value=8.,min_value=.1); finish(module,tool,{"G_Pa":G,"d_m":d,"D_m":D,"ActiveCoils":n},{"SpringRate_N_m":spring_rate(G,d,D,n)},r"k=\frac{Gd^4}{8D^3n}")
    elif tool=="Shaft Diameter from Torsion":
        T=st.number_input("Torque T (N·m)",value=500.,min_value=0.); tau=st.number_input("Allowable shear stress τ (MPa)",value=50.,min_value=.001); d=shaft_diameter_torsion(T,tau*1e6); finish(module,tool,{"Torque_Nm":T,"AllowableShearStress_MPa":tau},{"RequiredDiameter_mm":d*1000},r"d=\sqrt[3]{\frac{16T}{\pi\tau_{allow}}}")
    elif tool=="Flywheel Inertia":
        E=st.number_input("Energy to be absorbed/released E (J)",value=500.,min_value=.001); n1=st.number_input("Initial RPM",value=1000.,min_value=.1); n2=st.number_input("Final RPM",value=900.,min_value=0.); w1=2*math.pi*n1/60; w2=2*math.pi*n2/60; finish(module,tool,{"Energy_J":E,"InitialRPM":n1,"FinalRPM":n2},{"RequiredInertia_kgm2":flywheel_inertia(E,w1,w2)},r"I=\frac{2E}{\omega_1^2-\omega_2^2}")
    else:
        T=st.number_input("Torque (N·m)",value=100.,min_value=0.); rpm=st.number_input("RPM",value=1500.,min_value=.1); p=power(T,rpm); finish(module,tool,{"Torque_Nm":T,"RPM":rpm},{"Power_kW":p/1000},r"P=\frac{2\pi NT}{60}")

elif module=="Fluids":
    tools=["Reynolds Number","Continuity / Pipe Velocity","Pressure ↔ Head","Bernoulli Velocity","Darcy-Weisbach Pressure Drop","Hydraulic Power","Pipe Flow Rate"]
    tool=st.selectbox("Calculation",tools); st.session_state.active_tool=tool
    if tool=="Reynolds Number":
        rho=st.number_input("Density (kg/m³)",value=1000.,min_value=.0001); v=st.number_input("Velocity (m/s)",value=2.,min_value=0.); D=st.number_input("Diameter (m)",value=.05,min_value=.0001); mu=st.number_input("Dynamic viscosity (Pa·s)",value=.001,min_value=1e-8,format="%.6f"); Re=reynolds(rho,v,D,mu); regime="Laminar" if Re<2300 else "Transitional" if Re<=4000 else "Turbulent"; finish(module,tool,{"rho":rho,"V_ms":v,"D_m":D,"mu_Pas":mu},{"Reynolds":Re,"Regime":regime},r"Re=\frac{\rho VD}{\mu}")
    elif tool=="Continuity / Pipe Velocity":
        Q=st.number_input("Flow Q (m³/s)",value=.002,min_value=0.); D=st.number_input("Diameter D (m)",value=.05,min_value=.0001); A=math.pi*D**2/4; finish(module,tool,{"Q_m3s":Q,"D_m":D},{"Area_m2":A,"Velocity_ms":velocity_from_flow(Q,D)},r"Q=AV\qquad A=\frac{\pi D^2}{4}")
    elif tool=="Pressure ↔ Head":
        rho=st.number_input("Density (kg/m³)",value=1000.,min_value=.0001); h=st.number_input("Head (m)",value=10.,min_value=0.); finish(module,tool,{"rho":rho,"Head_m":h},{"Pressure_Pa":pressure_from_head(rho,G0,h)},r"P=\rho gh")
    elif tool=="Bernoulli Velocity":
        dp=st.number_input("Pressure difference ΔP (Pa)",value=5000.,min_value=0.); rho=st.number_input("Density (kg/m³)",value=1000.,min_value=.0001); finish(module,tool,{"DeltaP_Pa":dp,"rho":rho},{"IdealVelocity_ms":bernoulli_velocity(dp,rho)},r"\Delta P=\frac12\rho V^2")
    elif tool=="Darcy-Weisbach Pressure Drop":
        f=st.number_input("Darcy friction factor f",value=.02,min_value=0.); L=st.number_input("Pipe length (m)",value=10.,min_value=.001); D=st.number_input("Diameter (m)",value=.05,min_value=.0001); rho=st.number_input("Density (kg/m³)",value=1000.,min_value=.0001); v=st.number_input("Velocity (m/s)",value=2.,min_value=0.); finish(module,tool,{"f":f,"L_m":L,"D_m":D,"rho":rho,"V_ms":v},{"PressureDrop_Pa":darcy_drop(f,L,D,rho,v)},r"\Delta P=f\frac{L}{D}\frac{\rho V^2}{2}")
    elif tool=="Hydraulic Power":
        rho=st.number_input("Density (kg/m³)",value=1000.,min_value=.0001); Q=st.number_input("Flow Q (m³/s)",value=.01,min_value=0.); H=st.number_input("Head H (m)",value=20.,min_value=0.); eta=st.number_input("Pump efficiency",value=.75,min_value=.01,max_value=1.); p=rho*G0*Q*H/eta; finish(module,tool,{"rho":rho,"Q_m3s":Q,"Head_m":H,"Efficiency":eta},{"InputPower_W":p},r"P_{in}=\frac{\rho gQH}{\eta}")
    else:
        D=st.number_input("Pipe diameter D (m)",value=.05,min_value=.0001); v=st.number_input("Mean velocity V (m/s)",value=2.,min_value=0.); finish(module,tool,{"D_m":D,"Velocity_ms":v},{"FlowRate_m3s":velocity_from_flow(1,D)*0+math.pi*D**2/4*v},r"Q=AV=\frac{\pi D^2}{4}V")

elif module=="Thermodynamics":
    tools=["Ideal Gas P-V-T","ΔU","ΔH","Entropy Change (constant cp)","Isentropic Temperature","Efficiency / COP","Heat Engine Work"]
    tool=st.selectbox("Calculation",tools); st.session_state.active_tool=tool
    if tool=="Ideal Gas P-V-T":
        mode=st.radio("Calculate",["Pressure","Temperature"],horizontal=True); R=st.number_input("Gas constant R (J/kg·K)",value=287.,min_value=.0001); m=st.number_input("Mass (kg)",value=1.,min_value=.0001); V=st.number_input("Volume (m³)",value=1.,min_value=.0001)
        if mode=="Pressure": T=st.number_input("Temperature (K)",value=300.,min_value=.0001); finish(module,tool,{"R":R,"m":m,"V":V,"T":T},{"Pressure_Pa":ideal_pressure(m,R,T,V)},r"PV=mRT")
        else: P=st.number_input("Pressure (Pa)",value=101325.,min_value=.0001); finish(module,tool,{"R":R,"m":m,"V":V,"P":P},{"Temperature_K":ideal_temperature(P,R,m,V)},r"T=\frac{PV}{mR}")
    elif tool=="ΔU":
        m=st.number_input("Mass (kg)",value=1.,min_value=0.); cv=st.number_input("cv (J/kg·K)",value=718.,min_value=0.); T1=st.number_input("T1 (K)",value=300.); T2=st.number_input("T2 (K)",value=500.); finish(module,tool,{"m":m,"cv":cv,"T1":T1,"T2":T2},{"DeltaU_J":delta_u(m,cv,T1,T2)},r"\Delta U=mc_v(T_2-T_1)")
    elif tool=="ΔH":
        m=st.number_input("Mass (kg)",value=1.,min_value=0.); cp=st.number_input("cp (J/kg·K)",value=1005.,min_value=0.); T1=st.number_input("T1 (K)",value=300.); T2=st.number_input("T2 (K)",value=500.); finish(module,tool,{"m":m,"cp":cp,"T1":T1,"T2":T2},{"DeltaH_J":delta_h(m,cp,T1,T2)},r"\Delta H=mc_p(T_2-T_1)")
    elif tool=="Entropy Change (constant cp)":
        m=st.number_input("Mass (kg)",value=1.,min_value=.001); cp=st.number_input("cp (J/kg·K)",value=1005.,min_value=.001); T1=st.number_input("T1 (K)",value=300.,min_value=.001); T2=st.number_input("T2 (K)",value=500.,min_value=.001); finish(module,tool,{"m":m,"cp":cp,"T1":T1,"T2":T2},{"DeltaS_J_per_K":entropy_change(m,cp,T1,T2)},r"\Delta S=mc_p\ln\left(\frac{T_2}{T_1}\right)")
    elif tool=="Isentropic Temperature":
        T1=st.number_input("T₁ (K)",value=300.,min_value=.001); p1=st.number_input("p₁ (Pa)",value=101325.,min_value=.001); p2=st.number_input("p₂ (Pa)",value=500000.,min_value=.001); gamma=st.number_input("γ",value=1.4,min_value=1.01); finish(module,tool,{"T1_K":T1,"p1_Pa":p1,"p2_Pa":p2,"gamma":gamma},{"T2_K":isentropic_T(T1,p1,p2,gamma)},r"\frac{T_2}{T_1}=\left(\frac{p_2}{p_1}\right)^{(\gamma-1)/\gamma}")
    elif tool=="Efficiency / COP":
        out=st.number_input("Useful output / cooling effect",value=2000.,min_value=0.); inp=st.number_input("Input work",value=500.,min_value=.0001); finish(module,tool,{"Output":out,"Input":inp},{"Efficiency_or_COP":efficiency(out,inp)},r"\eta\;or\;COP=\frac{\text{useful output or cooling effect}}{\text{input work}}")
    else:
        Qin=st.number_input("Heat supplied Q_in (J)",value=10000.,min_value=0.); W=st.number_input("Net work W (J)",value=3000.,min_value=0.); finish(module,tool,{"Qin_J":Qin,"Work_J":W},{"Efficiency":W/Qin},r"\eta=\frac{W_{net}}{Q_{in}}")

elif module=="Heat Transfer":
    tools=["Conduction","Convection","Thermal Expansion","Overall Heat Transfer","Radiation","Thermal Resistance"]
    tool=st.selectbox("Calculation",tools); st.session_state.active_tool=tool
    if tool=="Conduction":
        k=st.number_input("Thermal conductivity k (W/m·K)",value=45.,min_value=.0001); A=st.number_input("Area A (m²)",value=1.,min_value=.0001); dt=st.number_input("Temperature difference ΔT (K)",value=50.); L=st.number_input("Wall thickness L (m)",value=.01,min_value=.0001); q=conduction(k,A,dt,L); finish(module,tool,{"k":k,"A_m2":A,"DeltaT_K":dt,"L_m":L},{"HeatRate_W":q,"HeatFlux_W_m2":q/A},r"\dot Q=\frac{kA\Delta T}{L}\qquad q''=\frac{\dot Q}{A}")
    elif tool=="Convection":
        h=st.number_input("h (W/m²·K)",value=25.,min_value=.0001); A=st.number_input("Area (m²)",value=1.,min_value=.0001); dt=st.number_input("ΔT (K)",value=30.); q=convection(h,A,dt); finish(module,tool,{"h":h,"A_m2":A,"DeltaT_K":dt},{"HeatRate_W":q,"HeatFlux_W_m2":q/A},r"\dot Q=hA\Delta T")
    elif tool=="Thermal Expansion":
        alpha=st.number_input("α (1/K)",value=12e-6,min_value=0.); L=st.number_input("Original length (m)",value=1.,min_value=0.); dt=st.number_input("ΔT (K)",value=50.); dl=thermal_expansion(alpha,L,dt); finish(module,tool,{"alpha":alpha,"L_m":L,"DeltaT_K":dt},{"Expansion_mm":dl*1000},r"\Delta L=\alpha L\Delta T")
    elif tool=="Overall Heat Transfer":
        U=st.number_input("Overall U (W/m²·K)",value=20.,min_value=.0001); A=st.number_input("Area (m²)",value=1.,min_value=.0001); dt=st.number_input("Overall ΔT (K)",value=40.); finish(module,tool,{"U":U,"A_m2":A,"DeltaT_K":dt},{"HeatRate_W":U*A*dt},r"\dot Q=UA\Delta T")
    elif tool=="Radiation":
        eps=st.number_input("Emissivity ε",value=.8,min_value=0.,max_value=1.); A=st.number_input("Area (m²)",value=1.,min_value=.0001); T1=st.number_input("Surface T₁ (K)",value=500.,min_value=0.); T2=st.number_input("Surrounding T₂ (K)",value=300.,min_value=0.); finish(module,tool,{"Emissivity":eps,"Area_m2":A,"T1_K":T1,"T2_K":T2},{"NetRadiation_W":radiation(eps,A,T1,T2)},r"\dot Q=\epsilon\sigma A(T_1^4-T_2^4)")
    else:
        L=st.number_input("Wall thickness L (m)",value=.01,min_value=.0001); k=st.number_input("k (W/m·K)",value=45.,min_value=.0001); A=st.number_input("Area (m²)",value=1.,min_value=.0001); finish(module,tool,{"L_m":L,"k":k,"A_m2":A},{"ConductionResistance_K_W":thermal_resistance(L,k,A)},r"R_{th}=\frac{L}{kA}")

elif module=="Engineering Math & Geometry":
    tools=["Circle","Cylinder","Cone","Sphere","Interpolation","2×2 Linear Equations","Polynomial Roots (Degree 2–6)"]
    tool=st.selectbox("Calculation",tools); st.session_state.active_tool=tool
    if tool=="Circle":
        D=st.number_input("Diameter (m)",value=.1,min_value=0.); finish(module,tool,{"Diameter_m":D},{"Area_m2":circle_area(D),"Circumference_m":math.pi*D},r"A=\frac{\pi D^2}{4}\qquad C=\pi D")
    elif tool=="Cylinder":
        D=st.number_input("Diameter (m)",value=.1,min_value=0.); L=st.number_input("Length (m)",value=.2,min_value=0.); finish(module,tool,{"Diameter_m":D,"Length_m":L},{"Volume_m3":cylinder_volume(D,L)},r"V=\frac{\pi D^2L}{4}")
    elif tool=="Cone":
        D=st.number_input("Diameter (m)",value=.1,min_value=0.); h=st.number_input("Height (m)",value=.2,min_value=0.); finish(module,tool,{"Diameter_m":D,"Height_m":h},{"Volume_m3":cone_volume(D,h)},r"V=\frac{\pi D^2h}{12}")
    elif tool=="Sphere":
        D=st.number_input("Diameter (m)",value=.1,min_value=0.); finish(module,tool,{"Diameter_m":D},{"Volume_m3":sphere_volume(D)},r"V=\frac{\pi D^3}{6}")
    elif tool=="Interpolation":
        x1=st.number_input("x₁",value=0.); x2=st.number_input("x₂",value=10.); y1=st.number_input("y₁",value=0.); y2=st.number_input("y₂",value=100.); x=st.number_input("Target x",value=5.); y=interpolation(x1,x2,y1,y2,x); finish(module,tool,{"x1":x1,"x2":x2,"y1":y1,"y2":y2,"x":x},{"InterpolatedY":y},r"y=y_1+\frac{(x-x_1)(y_2-y_1)}{x_2-x_1}")
    elif tool=="2×2 Linear Equations":
        st.caption("Solve a₁x+b₁y=c₁ and a₂x+b₂y=c₂."); a1=st.number_input("a₁",value=1.); b1=st.number_input("b₁",value=1.); c1=st.number_input("c₁",value=2.); a2=st.number_input("a₂",value=1.); b2=st.number_input("b₂",value=-1.); c2=st.number_input("c₂",value=0.); x,y=solve_linear2(a1,b1,c1,a2,b2,c2); finish(module,tool,{"a1":a1,"b1":b1,"c1":c1,"a2":a2,"b2":b2,"c2":c2},{"x":x,"y":y},r"a_1x+b_1y=c_1\qquad a_2x+b_2y=c_2")
    else:
        degree=st.selectbox("Polynomial degree",[2,3,4,5,6]); coeffs=[]
        for p in range(degree,-1,-1): coeffs.append(st.number_input(f"Coefficient of x^{p}" if p>1 else ("Coefficient of x" if p==1 else "Constant"),value=1.0 if p==degree else 0.0,key=f"coef_{degree}_{p}"))
        roots=polynomial_roots(coeffs); out={f"Root_{i+1}":complex(r) if abs(r.imag)>1e-9 else float(r.real) for i,r in enumerate(roots)}; finish(module,tool,{"Degree":degree,"Coefficients":", ".join(map(str,coeffs))},out,r"a_nx^n+a_{n-1}x^{n-1}+\cdots+a_1x+a_0=0")

elif module=="Manufacturing & Production":
    tools=["Cutting Speed ↔ RPM","Milling Feed Rate","Machining Time","Turning Material Removal Rate","Milling Material Removal Rate","Casting Shrinkage","Machining Power from Cutting Force","Spindle Speed from Cutting Speed"]
    tool=st.selectbox("Calculation",tools); st.session_state.active_tool=tool
    if tool=="Cutting Speed ↔ RPM":
        Vc=st.number_input("Cutting speed Vc (m/min)",value=100.,min_value=0.); D=st.number_input("Tool diameter D (mm)",value=10.,min_value=.001); finish(module,tool,{"Vc_m_min":Vc,"D_mm":D},{"RPM":cutting_rpm(Vc,D)},r"N=\frac{1000V_c}{\pi D}")
    elif tool=="Milling Feed Rate":
        fz=st.number_input("Feed per tooth fz (mm/tooth)",value=.1,min_value=0.); z=st.number_input("Number of teeth z",value=4.,min_value=.1); N=st.number_input("Spindle speed N (RPM)",value=1000.,min_value=0.); finish(module,tool,{"FeedPerTooth_mm":fz,"Teeth":z,"RPM":N},{"FeedRate_mm_min":milling_feed(fz,z,N)},r"V_f=f_z zN")
    elif tool=="Machining Time":
        L=st.number_input("Cutting length L (mm)",value=100.,min_value=.001); f=st.number_input("Feed per revolution f (mm/rev)",value=.2,min_value=.0001); N=st.number_input("Spindle speed N (RPM)",value=1000.,min_value=.1); t=machining_time(L,f,N); finish(module,tool,{"Length_mm":L,"Feed_mm_rev":f,"RPM":N},{"MachiningTime_min":t},r"t=\frac{L}{fN}")
    elif tool=="Turning Material Removal Rate":
        D=st.number_input("Work diameter D (mm)",value=50.,min_value=.001); d=st.number_input("Depth of cut d (mm)",value=2.,min_value=.001); f=st.number_input("Feed f (mm/rev)",value=.2,min_value=.0001); N=st.number_input("RPM",value=1000.,min_value=0.); finish(module,tool,{"D_mm":D,"Depth_mm":d,"Feed_mm_rev":f,"RPM":N},{"MRR_mm3_min":turning_mrr(D,d,f,N)},r"MRR=\pi D d f N")
    elif tool=="Milling Material Removal Rate":
        w=st.number_input("Width of cut w (mm)",value=10.,min_value=.001); ap=st.number_input("Axial depth ap (mm)",value=2.,min_value=.001); fz=st.number_input("Feed/tooth (mm/tooth)",value=.1,min_value=.0001); z=st.number_input("Teeth",value=4.,min_value=.1); N=st.number_input("RPM",value=1000.,min_value=0.); finish(module,tool,{"Width_mm":w,"Depth_mm":ap,"FeedPerTooth_mm":fz,"Teeth":z,"RPM":N},{"MRR_mm3_min":milling_mrr(w,ap,fz,z,N)},r"MRR=w\,a_p\,f_z\,z\,N")
    elif tool=="Casting Shrinkage":
        Lp=st.number_input("Pattern dimension Lp (mm)",value=100.,min_value=.001); Lc=st.number_input("Casting dimension Lc (mm)",value=99.,min_value=.001); finish(module,tool,{"Pattern_mm":Lp,"Casting_mm":Lc},{"LinearShrinkage_percent":casting_shrinkage(Lp,Lc)},r"Shrinkage(\%)=\frac{L_p-L_c}{L_c}\times100")
    elif tool=="Machining Power from Cutting Force":
        F=st.number_input("Cutting force F (N)",value=500.,min_value=0.); V=st.number_input("Cutting speed V (m/s)",value=1.,min_value=0.); finish(module,tool,{"Force_N":F,"Speed_ms":V},{"CuttingPower_W":F*V,"CuttingPower_kW":F*V/1000},r"P_c=FV")
    else:
        Vc=st.number_input("Cutting speed Vc (m/min)",value=100.,min_value=0.); D=st.number_input("Diameter D (mm)",value=10.,min_value=.001); finish(module,tool,{"Vc_m_min":Vc,"D_mm":D},{"SpindleRPM":cutting_rpm(Vc,D)},r"N=\frac{1000V_c}{\pi D}")

elif module=="Robotics & Vehicle":
    tools=["Wheel RPM","Wheel / Motor Torque","Motor Power","Robot Acceleration Force","Suspension Spring Rate","Suspension Natural Frequency","Chassis Bending Stress","Chassis Torsional Stress","Brake Disc Torque","Brake Disc Energy","Basic Ackermann Geometry"]
    tool=st.selectbox("Calculation",tools); st.session_state.active_tool=tool
    if tool=="Wheel RPM":
        v=st.number_input("Vehicle speed (m/s)",value=2.,min_value=0.); D=st.number_input("Wheel diameter (m)",value=.3,min_value=.001); finish(module,tool,{"Speed_ms":v,"WheelDiameter_m":D},{"WheelRPM":wheel_rpm(v,D)},r"N=\frac{60v}{\pi D}")
    elif tool=="Wheel / Motor Torque":
        mass=st.number_input("Robot mass (kg)",value=50.,min_value=0.); acc=st.number_input("Acceleration (m/s²)",value=1.,min_value=0.); radius=st.number_input("Wheel radius (m)",value=.15,min_value=.001); crr=st.number_input("Rolling resistance coefficient",value=.02,min_value=0.); grade=st.number_input("Road grade angle (deg)",value=0.); F,T=robot_force(mass,acc,crr,radius,grade); finish(module,tool,{"Mass_kg":mass,"Acceleration_ms2":acc,"WheelRadius_m":radius,"Crr":crr,"Grade_deg":grade},{"TractiveForce_N":F,"WheelTorque_Nm":T},r"F=ma+C_{rr}mg\cos\theta+mg\sin\theta\qquad T=Fr")
    elif tool=="Motor Power":
        T=st.number_input("Motor torque (N·m)",value=10.,min_value=0.); rpm=st.number_input("Motor RPM",value=1500.,min_value=0.); p=power(T,rpm); finish(module,tool,{"Torque_Nm":T,"RPM":rpm},{"MechanicalPower_W":p,"MechanicalPower_kW":p/1000},r"P=T\omega=T\frac{2\pi N}{60}")
    elif tool=="Robot Acceleration Force":
        m=st.number_input("Mass (kg)",value=50.,min_value=0.); a=st.number_input("Acceleration (m/s²)",value=1.,min_value=0.); finish(module,tool,{"Mass_kg":m,"Acceleration_ms2":a},{"InertialForce_N":m*a},r"F=ma")
    elif tool=="Suspension Spring Rate":
        target=st.number_input("Target wheel load (N)",value=2000.,min_value=.001); defl=st.number_input("Target suspension deflection (mm)",value=50.,min_value=.001); k=target/(defl/1000); finish(module,tool,{"WheelLoad_N":target,"Deflection_mm":defl},{"RequiredWheelRate_N_mm":k/1000,"RequiredWheelRate_N_m":k},r"k=\frac{F}{\delta}")
    elif tool=="Suspension Natural Frequency":
        k=st.number_input("Wheel/suspension stiffness k (N/m)",value=30000.,min_value=.001); sprung=st.number_input("Sprung mass per corner (kg)",value=50.,min_value=.001); finish(module,tool,{"Stiffness_N_m":k,"SprungMass_kg":sprung},{"NaturalFrequency_Hz":suspension_frequency(k,sprung)},r"f_n=\frac{1}{2\pi}\sqrt{\frac{k}{m_s}}")
    elif tool=="Chassis Bending Stress":
        M=st.number_input("Bending moment (N·m)",value=5000.,min_value=0.); c=st.number_input("Extreme-fibre distance (mm)",value=25.,min_value=.001); I=st.number_input("Second moment (mm⁴)",value=4e6,min_value=.001); s=chassis_bending_stress(M,c/1000,I*1e-12); finish(module,tool,{"Moment_Nm":M,"c_mm":c,"I_mm4":I},{"BendingStress_MPa":s/1e6},r"\sigma_b=\frac{Mc}{I}")
    elif tool=="Chassis Torsional Stress":
        T=st.number_input("Torsional moment (N·m)",value=2000.,min_value=0.); r0=st.number_input("Outer radius (mm)",value=25.,min_value=.001); J=st.number_input("Polar moment J (mm⁴)",value=8e6,min_value=.001); tau=chassis_torsion_stress(T,r0/1000,J*1e-12); finish(module,tool,{"Torque_Nm":T,"OuterRadius_mm":r0,"PolarMoment_mm4":J},{"TorsionalStress_MPa":tau/1e6},r"\tau=\frac{Tr}{J}")
    elif tool=="Brake Disc Torque":
        mu=st.number_input("Pad-disc friction coefficient μ",value=.35,min_value=0.); P=st.number_input("Clamp force / normal force (N)",value=5000.,min_value=0.); re=st.number_input("Effective friction radius (m)",value=.1,min_value=.0001); finish(module,tool,{"FrictionCoefficient":mu,"NormalForce_N":P,"EffectiveRadius_m":re},{"BrakingTorque_Nm":brake_torque(mu,P,re)},r"T_b=\mu F_N r_{eff}")
    elif tool=="Brake Disc Energy":
        m=st.number_input("Vehicle mass (kg)",value=300.,min_value=.001); v1=st.number_input("Initial speed (m/s)",value=20.,min_value=0.); v2=st.number_input("Final speed (m/s)",value=0.,min_value=0.); finish(module,tool,{"Mass_kg":m,"InitialSpeed_ms":v1,"FinalSpeed_ms":v2},{"KineticEnergyRemoved_J":brake_disc_energy(m,v1,v2)},r"\Delta E=\frac12m(v_1^2-v_2^2)")
    else:
        L=st.number_input("Wheelbase L (mm)",value=1200.,min_value=.1); T=st.number_input("Track width T (mm)",value=1200.,min_value=.1); R=st.number_input("Turn radius to vehicle centre R (mm)",value=3000.,min_value=.1);
        if R<=T/2: st.error("Turn radius must be greater than half the track width."); st.stop()
        inner=math.degrees(math.atan(L/(R-T/2))); outer=math.degrees(math.atan(L/(R+T/2))); finish(module,tool,{"Wheelbase_mm":L,"TrackWidth_mm":T,"CentreRadius_mm":R},{"InnerAngle_deg":inner,"OuterAngle_deg":outer},r"\tan\delta_i=\frac{L}{R-T/2}\qquad \tan\delta_o=\frac{L}{R+T/2}")

else:
    category=st.selectbox("Category",["Length","Force","Pressure","Power","Torque","Temperature"]); value=st.number_input("Value",value=1.); units={"Length":["m","mm","cm","inch","ft"],"Force":["N","kN","lbf"],"Pressure":["Pa","kPa","MPa","bar","psi"],"Power":["W","kW","hp"],"Torque":["N·m","N·mm","lbf·ft"],"Temperature":["°C","K","°F"]}; unit=st.selectbox("From",units[category]); r=unit_convert(category,value,unit); st.session_state.active_tool=category; finish(module,category,{"Value":value,"From":unit},r,"Standard SI-based unit conversion equations")

# -------------------------
# Active tool history
# -------------------------
active_tool=get_active_tool(module)
if module!="Home" and active_tool:
    active_df=history_dataframe(module,active_tool); st.divider(); st.subheader(f"📊 {active_tool} History")
    if not active_df.empty:
        st.dataframe(active_df,use_container_width=True,hide_index=True); c1,c2=st.columns(2)
        with c1: st.download_button("⬇️ Download This History as Excel",excel_for_history(module,active_tool),file_name=f"{module.replace(' ','_')}_{active_tool.replace(' ','_')}_history.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
        with c2:
            if st.button("🗑️ Delete This Tool History",use_container_width=True):
                st.session_state.history=[x for x in st.session_state.history if not (x["Module"]==module and x["Tool"]==active_tool)]; st.session_state.last_history_signature=None; st.rerun()
        st.caption(f"{len(active_df)} saved calculation(s) for this tool. No timestamp and no unrelated module data.")
    else: st.caption("No saved calculations for this tool yet.")
