%% init_env
% Description:
%   Configures graphics properties, LaTeX properties, etc.
%   Should be called early in the main function.
% Inputs:
%   none
% Outputs:
%   none

function init_env
    set(groot,'defaultLegendInterpreter','latex');
    set(groot,'defaultTextInterpreter','latex');
    set(groot,'defaultAxesTickLabelInterpreter','latex');
    set(groot,'DefaultAxesFontSize',16); % make text large enough to read
end