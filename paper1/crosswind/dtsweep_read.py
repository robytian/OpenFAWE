import json,os,numpy as np,sys
O=sys.argv[1] if len(sys.argv)>1 else '.'
O=O.rstrip('/')+'/'
AREA,CH,RHO=32.9285,1.2831,1.225
def load(name):
    p=O+name+'/checks.json'
    if not os.path.exists(p): return None
    d=json.load(open(p)); a=d['per_step']['A']; dt=d['settings']['dt_s']
    st=np.array(a['step']); f=np.array(a['force_body_N']); m=np.array(a['moment_origin_body_Nm']); v=np.array(a['airspeed_m_s'])
    q=0.5*RHO*v*v
    return dict(t=st*dt, CZ=f[:,2]/(q*AREA), CX=f[:,0]/(q*AREA), Cm=m[:,1]/(q*AREA*CH), dt=dt)
rows=[]
for n in (468,936,1872,3744,7488):
    r,w=load(f'r{n}'),load(f'n{n}')
    if r is None or w is None: continue
    out={'dt':r['dt']}
    for k in ('CZ','Cm','CX'):
        e=r[k]-w[k]
        for T in (0.25,0.5,1.0):
            out[f'{k}@{T}']=float(np.interp(T,r['t'],e))
        sel=(r['t']>0.5)&(r['t']<=1.0+1e-9); out[f'{k} mean 0.5-1 s']=float(e[sel].mean())
    rows.append(out)
keys=[k for k in rows[0] if k!='dt']
print('dt     '+'  '.join(f'{k:>16}' for k in keys))
for o in rows: print(f"{o['dt']:.4f} "+'  '.join(f'{o[k]:16.5f}' for k in keys))
json.dump(rows,open(O+'summary.json','w'),indent=1)
