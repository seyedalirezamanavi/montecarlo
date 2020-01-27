"""
credit: Seyed Alireza Manavi
gmail: seyedalirezamanavi1402@gmail.com
github: https://github.com/seyedalirezamanavi


"""



import numpy as np
import matplotlib.pyplot as plt


def total_energy(lattic,boundarycondition):
    if boundarycondition=="periodic":
        lattic_up = np.roll(lattic,-1,axis = 0)
        lattic_down = np.roll(lattic,1,axis = 0)
        lattic_left = np.roll(lattic,-1,axis = 1)
        lattic_right = np.roll(lattic,1,axis = 1)
        Energy = np.multiply(lattic,(lattic_up + lattic_down + lattic_left + lattic_right))
        return -np.mean(Energy), np.mean(Energy**2), np.mean(Energy)**2
    else:
        lattic_up = np.multiply(lattic,np.roll(lattic,-1,axis = 0))
        lattic_up[0,:] = 0
        lattic_down = np.multiply(lattic,np.roll(lattic,1,axis = 0))
        lattic_down[-1,:] = 0
        lattic_left = np.multiply(lattic,np.roll(lattic,-1,axis = 1))
        lattic_left[:,-1] = 0
        lattic_right = np.multiply(lattic,np.roll(lattic,1,axis = 1))
        lattic_right[:,0] = 0
        Energy = (lattic_up + lattic_down + lattic_left + lattic_right)
        return -np.mean(Energy), np.mean(Energy**2), np.mean(Energy)**2

def neighbors(x, y, lattic):
    lx = np.shape(lattic)[0]
    ly = np.shape(lattic)[1]
    f = [0,0,0,0]
    if (y + 1) % ly != y + 1:
        f[0] = 1
    if (x + 1) % lx != x + 1:
        f[1] = 1
    if (y - 1) < 0:
        f[2] = 1
    if (x - 1) < 0 :
        f[3] = 1
    return f

def energy(x, y, lattic, boundarycondition):
    lx = np.shape(lattic)[0]
    ly = np.shape(lattic)[1]
    if boundarycondition == "periodic":
        E = lattic[x,y] * (lattic[x,(y+1)%ly] + lattic[(x+1)%lx,y] + lattic[x-1,y] + lattic[x,y-1])
    else:
        if (y+1)%ly != y+1 and (x + 1)%lx != x + 1 and (y-1) < 0 and (x-1) < 0 :
            f = neighbors(x, y, lattic)
            E = lattic[x,y] * (f[0]*lattic[x,(y+1)%ly] + f[1]*lattic[(x+1)%lx,y] + f[3]*lattic[x-1,y] + f[2]*lattic[x,y-1])
        else:
            E = lattic[x,y] * (lattic[x,(y+1)%ly] + lattic[(x+1)%lx,y] + lattic[x-1,y] + lattic[x,y-1])
    return -E

def local_update(x, y, lattic, boundarycondition, beta):
    E_unfliped = energy(x, y, lattic, boundarycondition)
    fliped_lattic = lattic.copy()
    fliped_lattic[x,y] = -fliped_lattic[x,y]
    E_fliped = energy(x, y, fliped_lattic, boundarycondition)
    p = np.exp(-beta * (E_fliped - E_unfliped))
    r = np.random.random()
    # print(r,p)
    if p>r :
        # print("accepted", r, p)
        lattic[x, y] = fliped_lattic[x, y]
    else:
        pass
        # print("rejected", r, p)
    return lattic

def magnetization(lattic):
    return np.sum(lattic)

def heat_capacity(lattic, beta, boundarycondition):
    conf_energy, E2_avr, E_avr2 = total_energy(lattic, boundarycondition)
    C = (E2_avr - E_avr2) * beta**2
    return conf_energy, C

def magnetic_susceptibility(beta, lattic):
    return (np.mean(lattic*lattic) - np.mean(lattic)**2) * beta


def plot_function(fig, title_string, x_arr, y_arr, ytitle):
    plt.cla()  # clear  axis
    plt.clf()  # clear  figure
    plt.plot(x_arr, y_arr, '.', linewidth=4, label=ytitle)
    plt.ylabel(ytitle, fontsize=16)
    plt.xlabel('T', fontsize=16)
    plt.legend(loc='best', fontsize='small')
    plt.title(title_string)
    # plt.grid()
    fig.savefig("plots/"+title_string+".png")
    # plt.show()


def main():
    beta = 0.1
    m = []
    c = []
    e = []
    chi = []
    T = []
    for j in range(100):
        beta += .01
        T.append(1/beta)
        boundarycondition = "open"
        lattic = np.power(-1, np.random.randint(2, size = (20,20)))
        for i in range(500):
            for x in range(np.shape(lattic)[0]):
                for y in range(np.shape(lattic)[1]):
                    local_update(x, y, lattic, boundarycondition, beta)

        ee, cc = heat_capacity(lattic,beta,boundarycondition)
        e.append(ee)
        c.append(cc)
        chi.append(magnetic_susceptibility(beta, lattic))
        m.append(magnetization(lattic))
        e_, c_ = heat_capacity(lattic,beta,boundarycondition)
        print("T: %f, total_energy: %f, heat_capacity: %f, magnetic_susceptibility: %f, magnetization: %f"%(1/beta, e_, c_, magnetic_susceptibility(beta, lattic), magnetization(lattic)))

    # print(np.argmax(c))
    fig = plt.figure()
    plot_function(fig,r"$<E>_T$",T,e,"Energy")
    plot_function(fig,r"$<\chi>_T$",T,chi,"Magnetic Susceptibility")
    plot_function(fig,r"$<M>_T$",T,m,"Magnetization")
    plot_function(fig,r"$<C>_T$",T,c,"Heat Capacity")
    # plt.imshow(lattic)
    # plt.show()
main()
