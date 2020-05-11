def moment(R,r,s):
    result = 0
    for i,j in R:
        result += (i**r*j**s)
    return result

def mu(R,r,s):
    m_00 = moment(R,0,0)
    m_01 = moment(R,0,1)
    m_10 = moment(R,1,0)
    avg_i = int(round(m_10/m_00))
    avg_j = int(round(m_01/m_00))
    mu = 0
    for i,j in R:
        mu += (i-avg_i)**r*(j-avg_j)**s
    return mu

def nu(R,r,s):
    t = (r+s)/2 + 1
    return mu(R,r,s)/(mu(R,0,0)**t)

def hu_1(R):
    return nu(R,2,0) + nu(R,0,2)

def hu_2(R):
    return (nu(R,2,0)-nu(R,0,2))**2 + 4*nu(R,1,1)**2

def hu_3(R):
    return (nu(R,3,0)- 3*nu(R,1,2))**2 + (3*nu(R,2,1) - nu(R,0,3))**2

def hus(R):
    return hu_1(R), hu_2(R), hu_3(R)