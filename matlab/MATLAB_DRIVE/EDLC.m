function EDLC_SDL_0_CV_SIGMA_2
%% ========================================================================
%  EDLC Modeling with finite thickness separator
%  Biesheuvel and Bazant (2010)
%  Khaled Hasan version
%  with Matrix Conduction
%  infinite thin separator
%% ========================================================================

clear all
close all

% variables:
    %--- Diffusivities
    Ds = 2e-4;  %--- diffusivity separator
    De = 2e-9;  %--- diffusivity electrode
    d_sdl = Ds/De;

    %--- Thicknesses
    Ls = 0.0001; %--- thickness separator
    Le = 0.050; %--- thickness electrode 
    lratio = Ls/Le;

    tau_e = Le^2/De;    
    tau_s = Ls^2/Ds;

    eps = 0.121;    %--- eps value
    p = 0.33333333;       %--- porosity of electrode
    SIGMA = 100.;      %--- conductivity matrix
    alpha_cm = 1;        %--- WITH (1) / Without (1) matrix conductivity

    beta  = p/d_sdl*lratio;
    % beta  = p/d_sdl;

    %--- time steps
    ntime = 1*10^8;   %--- number of time steps
    Dt = 1e-6;
    % Dtcrit = 10^-4 * tau_s/tau_e;
    % if Dt > Dtcrit
    %     Dt = Dtcrit
    % end

    Time = Dt*cumsum(ones(ntime,1));

    %--- length discretization (number of grid points)
    Nx = 101;      
    dx = 1/(Nx-1);

    %--- cyclic charge input
    tau_ratio = 0.008;      %--- ramp time --to-- diffusion time ratio
    phi_lim = - 39.0;      %--- maximum voltage
    u = -phi_lim/tau_ratio;          %--- voltage rate

    phi1  = zeros(ntime,1);  %--- Matrix Voltage at (x=1,t)
    phi_1 = -Dt*ones(Nx,1);  %--- Matrix Voltage (along x)
    dphi_1= zeros(Nx,1);     %--- Increment of matrix voltage (along x) 

%--- initial conditions
%   --- in electrode
    c       = ones(Nx,1);       %--- pore concentration (along x)
    phi     = zeros(Nx,1) + phi_1;  %--- Pore voltage (along x)
    ie      = zeros(Nx,1);      %--- pore current (along x)
    im_1    = zeros(Nx,1);      %--- matrix current (along x)
    im1     = zeros(ntime,1);   %--- matrix current (at x=1,t)
    ie1     = zeros(ntime,1);   %--- pore current at (x=0,t)


%   --- interface 
    phi_int = zeros(ntime,1);   %--- pore voltage at (x=0,t)
    phi_int(1) = phi_1(1);      %------ initial value
    phi_end = zeros(ntime,1);   %--- pore voltage at (x=1,t)
    phi_end(1) = phi_1(end);    %------ intial value
    c_int = zeros(ntime,1);     %--- concentration at (x=0,t)
    c_int(1) = c(1);            %------ initial value
    c_end = zeros(ntime,1);     %--- concentration at (x=1,t)
    c_end(1) = c(end);          %------ initial value

    nprint = 1;
    icycle = 0;
    n_cap  = 0;

    sign = -1;

for itime = 2:ntime
    dphidt = zeros(Nx,1);
    dcdt   = zeros(Nx,1);
    dt = Dt;

    %--- CV-Loading
    %----- Check charge-discharge voltage
    if (sign == -1 && phi1(itime-1) <= phi_lim) || (sign == 1 && phi1(itime-1) >= 0)
        sign = sign * -1;
        nprint = 1;
        n_cycle(icycle) = itime-1;
    end

    %----- Update Matrix Current
    phi_1_0 = phi_1;
    dphi1 = sign * u*dt;                  %--- increment at x = 1
    phi1(itime) = phi1(itime-1) + dphi1;  %--- total at x = 1 

    phi1_0 = phi1(itime) - alpha_cm* p/(1-p) / SIGMA * sum(ie-ie(1)) * dx; %--- phi_1(0)
    phi_1 = phi1_0 + alpha_cm* p/(1-p) / SIGMA * cumsum(ie-ie(1)) * dx;    %--- phi_1(x)
    dphi_1 = phi_1 - phi_1_0;             %--- increment for timestep

   
    %--- ELECTRODE
    %--- solution from previous time step
    c0 = c;
    phi0 = phi;

    A = d2dx2(c0,dx);
    B = ddx(phi0,dx).*ddx(c0,dx) + c0.*d2dx2(phi0,dx);

    e1 = 1 + 2*eps*(sinh((phi0-phi_1)/4).^2).*(sqrt(c0)>0).^(-1);
    e2 = eps*(sqrt(c0)>0).*(sinh((phi0-phi_1)/2));
    e3 = eps*((sqrt(c0)>0).^(-1)).*(sinh((phi0-phi_1)/2));
    e4 = eps*(sqrt(c0)>0).*(cosh((phi0-phi_1)/2));

    denom = e1.*e4 - e2.*e3;
    j_min = find(min(denom),1);

    if denom(j_min) > 0.01
    
    %--- update (c,phi) in electrode
        dcdt = (A.*e4 - B.*e2).*((denom).^(-1));
        dphidt = (B.*e1 - A.*e3).*((denom).^(-1)) + dphi_1/dt ;
        % dphidt = (B.*e1 - A.*e3).*((denom).^(-1)) + dphi1/dt ;

        phi = phi0 + dt*(dphidt);
        c   = c0   + dt*dcdt;
    else
        keyboard
    end


    %--- Applying boundary conditions 
    %   --- at x=1
    c(end) = c(end-1);
    phi(end) = phi(end-1);

    %   --- at x=0 (separator)
    c(1)    = c(2);
    fac     = 1/(1+dx/beta);
    phi(1)  = fac * phi(2);

    %--- CURRENT update
    ie = - c.*ddx(phi,dx);
    ie(1) = - c(1)*(phi(2)-phi(1))/dx;

    im_1 = p/(1-p) * (ie(1) - ie);

    %--- store for display
    c_int(itime)    = c(1);
    c_end(itime)    = c(end);
    phi_int(itime)  = phi(1);
    phi_end(itime)  = phi(end);
    im1(itime)      = im_1(end);

    if nprint == 1
        icycle = icycle + 1
        s = (-1)^icycle;

        figure(1)
        subplot(2,2,1)
            plot(c)        
            xlabel('Position $x$',FontSize=12,Interpreter='latex')
            ylabel('Concentration $c$',FontSize=12,Interpreter='latex')
            hold on
        subplot(2,2,2)
            plot(phi)
            xlabel('Position $x$',FontSize=12,Interpreter='latex')
            ylabel('Pore potential $\phi$',FontSize=12,Interpreter='latex')
            hold on
        subplot(2,2,3)
            plot(Time(1:itime),c_int(1:itime))
            hold on
            plot(Time(1:itime),c_end(1:itime))
            xlabel('Time $t$',FontSize=12,Interpreter='latex')
            ylabel('$c_{int/end}$',FontSize=12,Interpreter='latex')
            hold off
        subplot(2,2,4)
            plot(Time(1:itime),phi_int(1:itime))
            hold on
            plot(Time(1:itime),phi_end(1:itime))
            xlabel('Time $t$',FontSize=12,Interpreter='latex')
            ylabel('$\phi_{int/end}$',FontSize=12,Interpreter='latex')
            hold off
            % hold on

        figure(2)   %--- matrix current at x=1 (current collector)
            subplot(2,2,1)
            plot(-phi1(1:itime),im1(1:itime))
            xlabel('Voltage $\phi_1$',Interpreter='latex')
            ylabel('Matrix Current $i_m(x=1)$',Interpreter='latex')

        nprint = 0;

        %--- Determine Capacitance
        if (icycle >= 5) && s == -1
            n_cap = n_cap + 1;
            i_start = n_cycle(icycle-3);
            i_end   = n_cycle(icycle-1);
            X = -phi1(i_start:i_end);
            Y = im1(i_start:i_end);
            scanrate = u;
            V_0 = -phi_lim;
            %--- Fit of CV-curve
                [t_c(n_cap),t_0(n_cap),Res(n_cap),Cap(n_cap)]= FIT_CV(X,Y,scanrate,V_0);
            %--- Fit of Area
                CV_area(n_cap) = CVarea(X,Y);
                Capp(n_cap) = (CV_area(n_cap))/(scanrate)/V_0/2;
                tcV(n_cap) = (t_c(n_cap)/t_0(n_cap));
                C_0app(n_cap) = Capp(n_cap)/(1-((tcV(n_cap))^(-1))*(2*(1-exp(-tcV(n_cap)))^2)*(1-exp(-2*tcV(n_cap))));
            figure(2)
                subplot(2,2,3)
                plot(Cap/(eps*p/(1-p)),'o--')
                hold on
                plot(C_0app/(eps*p/(1-p)),'o--')
                legend('From CV-Fit','From CV-Area','Interpreter','latex','Location','best')
                xlabel('Cycle','Interpreter','latex')
                ylabel('Capacitance$/\epsilon*\frac{p}{1-p}$','Interpreter','latex')
                hold off

                subplot(2,2,4)
                plot(t_0,'o--')
                xlabel('Cycle','Interpreter','latex')
                ylabel('Time $\tau_0/\tau_e=CR/\tau_e$','Interpreter','latex')
        end
    end
end

    keyboard

return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ddx = ddx(arr,dx)
%==========================================================================
N = length(arr);
ddx =zeros(N,1);
ddx(2:N-1) = (arr(3:N)-arr(1:N-2))/2/dx;
return
end

%==========================================================================
function d2dx2 = d2dx2(arr,dx)
%==========================================================================
N = length(arr);
d2dx2 =zeros(N,1);
d2dx2(2:N-1) = (arr(3:N)-2*arr(2:N-1)+arr(1:N-2))/dx/dx;
return
end

%==========================================================================
%% ========================================================================
function [tau_c,tau_0,R_0,C_0]= FIT_CV(X,Y,scanrate,V_0)
%
% CV-FIT
%% ========================================================================

N  = length(X);
iX = zeros(N,1);

tau_c = V_0/scanrate;

%--- 1. iDENTIFY CHARGE - DISCHARGE
Xdata = X/V_0;
Ydata = Y;
iX(1) = 1;
k = 0;
for i = 2:N
    if (X(i) > X(i-1)) && (Y(i) > Y(i-1))  %--- Charge
        iX(i) =  1;
    elseif (X(i) < X(i-1)) && (Y(i) < Y(i-1)) %--- Discharge
        iX(i) = -1;
        k = k + 1;
        Xsym(k) = Xdata(i);
        Ysym(k) = Ydata(i);
        iXsym(k)= -1;
    else
        iX(i) = iX(i-1);
    end
end
%--- "Symmetrization": use discharge curve for charge curve
for ii = k+1:2*k
    Xsym(ii) = 1-Xsym(ii-k);
    Ysym(ii) = - Ysym(ii-k);
    iXsym(ii)= 1;
end
%    --- correct for Faradic effects in charge curve
minYsymD = min(Ysym(1:k));
maxYsymC = max(Ysym(k+1:2*k));
Xsym2  = Xsym;
iXsym2 = iXsym;
Ysym2(1:k)      = min(Ysym(1:k),maxYsymC);
Ysym2(k+1:2*k)  = max(Ysym(k+1:2*k),minYsymD);

%--- 2. MODEL-FIT of CV-curve (steady-state conditions)
Xdata1  =(-iX.*(Xdata-1/2*(1-iX)));
Xdatasym=(-iXsym.*(Xsym-1/2*(1-iXsym)));
Xdatasym2=(-iXsym2.*(Xsym2-1/2*(1-iXsym2)));

%----- x(1)=I_0=V_0/R_0 | x(2) = tau_0/tau_c

fun_c0 = @(x)(x(2))*(iXsym-2*iXsym*(1-exp(-1/(x(2))))/(1-exp(-2/(x(2)))).*exp(Xdatasym/(x(2))))-Ysym/(x(1));
% fun_c  = @(x)(x(2))*(iXsym2-2*iXsym2*(1-exp(-1/(x(2))))/(1-exp(-2/(x(2)))).*exp(Xdatasym2/(x(2))))-Ysym2/(x(1));
fun_c = @(x)(x(2))*(iX-2*iX*(1-exp(-1/(x(2))))/(1-exp(-2/(x(2)))).*exp(Xdata1/(x(2))))-Ydata/(x(1));
x0 = [max(Ydata)/10,.5];
x = lsqnonlin(fun_c,x0,[0.01,0.00001],[50*max(Ydata),1000]);
x_0= lsqnonlin(fun_c0,x0,[0.01,0.001],[50*max(Ydata),1000]);

%----- Output Parameter calculations 
tau_0 = x(2) * tau_c;
R_0 = V_0/x(1); 
C_0 = tau_0/R_0;

%----- Plot
Ymod = x(1)*x(2)*(iX-2*iX*(1-exp(-1/x(2)))/(1-exp(-2/x(2))).*exp(Xdata1/x(2)));

figure(2)
subplot(2,2,2)
    plot(Xdata,Ydata,'.',Xdata,Ymod,'lineWidth',.5)
    hold on
    grid
    xlabel('Voltage $\phi_1/\max\phi_1$','Interpreter','latex')
    ylabel('Matrix Current $i_m$','Interpreter','latex')
    hold off

return
end

%% ========================================================================
function OINT = CVarea(X,Y)
%
% CV Area claculation
% (for the entire CV curve)
%% ========================================================================

N = length(X);
iX = zeros(N,1);
%--- 1. IDENTIFY CHARGE - DISCHARGE
Xdata = X;
Ydata = Y;
iX(1) = 1;
Ndischarge = N;
for i = 2:N
    if (X(i) > X(i-1)) && (Y(i) > Y(i-1))  %--- Charge
        iX(i) =  1;
    elseif (X(i) < X(i-1)) && (Y(i) < Y(i-1)) %--- Discharge
        iX(i) = -1;
    else
        iX(i) = iX(i-1);
    end
    if iX(i)==-1 && iX(i-1)==1 % --- switch from charge to discharge
        Ncharge = i-1;
    elseif iX(i)==1 && iX(i-1)==-1
        Ndischarge = i-1;      % --- switch from discharge to charge
    end
end

%--- 2. INTEGRAL
OINT_charge = 0.;
for i = 2:Ncharge
    OINT_charge = OINT_charge + (Y(i)+Y(i-1))/2 * iX(i)*(X(i)-X(i-1));
end
OINT_discharge = 0.;
for i = Ncharge+2:Ndischarge
    OINT_discharge = OINT_discharge - (Y(i)+Y(i-1))/2 * iX(i)*(X(i)-X(i-1));
end

Ratio = OINT_charge/OINT_discharge;
if Ratio > 1.05
    OINT = 2*OINT_discharge;
else
    OINT = OINT_charge + OINT_discharge;
end
% keyboard

return
end
