%GEN_STATS_FIXTURES  Regenerate MATLAB parity fixtures for pydynamo.stats
%
%   Usage:
%       run('scripts/gen_stats_fixtures.m')
%
%   Inputs:
%       none
%
%   Outputs:
%       none (writes tests/fixtures/stats_*.csv)
%
%   Notes:
%       Produces reference values for tests/test_stats_parity.py. Cases
%       deliberately span the places where a naive port drifts: both
%       sides of ranksum's exact/approximate switch (n1+n2 = 20) and
%       signrank's (n = 15), heavy/light/no ties, samples containing
%       NaN, and FDR families containing NaN p-values.
%
%       A diff in the regenerated fixtures means MATLAB moved -- treat
%       that as a decision, not a file to commit.
%
%   See also: ranksum, signrank, FDR_1D, FDR_2D
%
%   ∿∿∿  Prerau Laboratory · sleepEEG.org  ∿∿∿

this_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(this_dir, '..', 'tests', 'fixtures');
assert(isfolder(out_dir), 'fixture dir not found: %s', out_dir);
addpath(genpath('/Users/Mike/code/toolboxes/DYNAM-O_toolbox/DYNAM-O_dev'));

%% ranksum
rng(2026);
f = fopen(fullfile(out_dir,'stats_ranksum.csv'),'w');
fprintf(f,'case,p,x,y\n'); n=0;
for trial = 1:250
    n1 = randi([1 25]); n2 = randi([1 25]);
    switch randi(4)
        case 1, x=randn(1,n1);          y=randn(1,n2)+0.6;
        case 2, x=randi([1 4],1,n1);    y=randi([1 4],1,n2);
        case 3, x=round(randn(1,n1),1); y=round(randn(1,n2)+0.3,1);
        case 4, x=[randn(1,max(1,n1-2)) nan(1,min(2,n1))];
                y=[randn(1,max(1,n2-2))+0.4 nan(1,min(2,n2))];
    end
    if sum(isfinite(x))==0 || sum(isfinite(y))==0, continue; end
    n=n+1;
    fprintf(f,'%d,%.17g,%s,%s\n', n, ranksum(x,y), vec2s(x), vec2s(y));
end
fclose(f); fprintf('ranksum: %d cases\n', n);

%% signrank
rng(777);
f = fopen(fullfile(out_dir,'stats_signrank.csv'),'w');
fprintf(f,'case,p,x,y\n'); n=0;
for trial = 1:250
    m = randi([2 30]);
    switch randi(4)
        case 1, x=randn(1,m);           y=randn(1,m)+0.5;
        case 2, x=randi([1 5],1,m);     y=randi([1 5],1,m);
        case 3, x=round(randn(1,m),1);  y=round(randn(1,m)+0.4,1);
        case 4, x=randn(1,m); y=randn(1,m)+0.5;
                k=randperm(m,max(1,floor(m/4))); x(k)=nan;
    end
    d = x-y; d = d(isfinite(d)); d = d(abs(d)>0);
    if isempty(d), continue; end
    n=n+1;
    fprintf(f,'%d,%.17g,%s,%s\n', n, signrank(x,y), vec2s(x), vec2s(y));
end
fclose(f); fprintf('signrank: %d cases\n', n);

%% fdr_bh (via FDR_1D's nested copy, exercised through FDR_1D itself)
rng(31);
f = fopen(fullfile(out_dir,'stats_fdr1d.csv'),'w');
fprintf(f,'case,method,q,nbins,ntrials1,ntrials2,g1,g2,p_values,p_adj,sigbins\n'); n=0;
for trial = 1:30
    R = randi([5 40]); N1 = randi([4 20]); N2 = randi([4 20]);
    g1 = randn(R,N1); g2 = randn(R,N2) + 0.8;
    if mod(trial,3)==0                       % sprinkle missingness
        g1(randi(R,1,max(1,floor(R/3))), :) = nan;
        for r = 1:R
            k = randperm(N2, randi([0 max(1,floor(N2/2))]));
            g2(r,k) = nan;
        end
    end
    for ms = {'dependent','independent'}
        q = 0.05;
        [sig, padj, pval] = FDR_1D(g1, g2, 'FDR', q, 'method', ms{1}, 'ploton', false);
        n = n+1;
        fprintf(f,'%d,%s,%.17g,%d,%d,%d,%s,%s,%s,%s,%s\n', n, ms{1}, q, R, N1, N2, ...
            vec2s(g1(:)'), vec2s(g2(:)'), vec2s(pval(:)'), vec2s(padj(:)'), vec2s(double(sig(:)')));
    end
end
fclose(f); fprintf('FDR_1D: %d cases\n', n);

%% FDR_2D — including the paired (signrank) path, which FDR_1D cases miss
rng(99);
f = fopen(fullfile(out_dir,'stats_fdr2d.csv'),'w');
fprintf(f,'case,method,paired,q,R,C,N1,N2,g1,g2,p_values,p_adj,sigbins\n'); n=0;
for trial = 1:12
    R = randi([3 8]); C = randi([3 8]); N = randi([6 16]);
    paired = mod(trial,2)==1;
    if paired, N1 = N; N2 = N; else, N1 = N; N2 = randi([6 16]); end
    g1 = randn(R,C,N1); g2 = randn(R,C,N2) + 0.7;
    if mod(trial,3)==0
        g1(1,:,:) = nan;                       % a fully missing row
        g2(2,1,1:max(1,floor(N2/2))) = nan;    % partial missingness
    end
    for ms = {'dependent','independent'}
        q = 0.05;
        [sig,padj,pval] = FDR_2D(g1,g2,'FDR',q,'method',ms{1}, ...
                                 'paired',paired,'ploton',false);
        n = n+1;
        fprintf(f,'%d,%s,%d,%.17g,%d,%d,%d,%d,%s,%s,%s,%s,%s\n', n, ms{1}, paired, q, ...
            R, C, N1, N2, vec2s(g1(:)'), vec2s(g2(:)'), ...
            vec2s(pval(:)'), vec2s(padj(:)'), vec2s(double(sig(:)')));
    end
end
fclose(f); fprintf('FDR_2D: %d cases\n', n);

%% gpermtest2 — RNG differs across languages, so record the DATA plus
%% MATLAB's observed statistic and bound for a distributional check.
rng(5150);
f = fopen(fullfile(out_dir,'stats_gpermtest2.csv'),'w');
fprintf(f,'case,alpha,iterations,R,C,N1,N2,g1,g2,true_stat,bounds,sigbins\n'); n=0;
for trial = 1:6
    R = randi([3 6]); C = randi([3 6]); N1 = randi([10 20]); N2 = randi([10 20]);
    g1 = randn(R,C,N1); g2 = randn(R,C,N2);
    if mod(trial,2)==0, g1(2,2,:) = g1(2,2,:) + 3; end   % planted effect
    alpha = 0.05; iters = 2000;
    [sig, bounds, tstat] = gpermtest2(g1,g2,alpha,@(x)mean(x,2,'omitnan'),iters,false);
    n = n+1;
    fprintf(f,'%d,%.17g,%d,%d,%d,%d,%d,%s,%s,%s,%s,%s\n', n, alpha, iters, R, C, N1, N2, ...
        vec2s(g1(:)'), vec2s(g2(:)'), vec2s(tstat(:)'), vec2s(bounds(:)'), vec2s(double(sig(:)')));
end
fclose(f); fprintf('gpermtest2: %d cases\n', n);

function s = vec2s(v)
    s = strjoin(arrayfun(@(x) sprintf('%.17g',x), v, 'uni', 0), ' ');
end
