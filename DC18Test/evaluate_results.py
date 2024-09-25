import os.path, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from examples.DC18_classes import DC18Answers
from plot_results import PlanetFitInfo





class AllResults():
    pspl_fit_types = ['Initial PSPL Guess', 'Initial SFIT', 'Revised SFIT']
    planet_fit_types = ['2L1S Guess']
    fit_types = [*pspl_fit_types, *planet_fit_types]
    _colors = ['black', 'magenta', 'limegreen', 'blue']

    def __init__(self, path='.'):
        self.results = self.get_results(path)
        self.answers = self.get_answers()

        self.colors = {AllResults.fit_types[i]: AllResults._colors[i] for i in range(len(AllResults.fit_types))}
        self._delta_t_0 = None
        self._delta_u_0 = None
        self._delta_t_E = None
        self._delta_rho = None
        self._delta_s = None
        self._delta_q = None
        self._delta_alpha = None

    def get_results(self, path):
        results = {}
        for key in AllResults.fit_types:
            results[key] = None

        logs = glob.glob(os.path.join(path, 'WFIRST*.log'))
        for file in np.sort(logs):
            planet = PlanetFitInfo(file)
            for fit_type, params in zip(
                    AllResults.fit_types,
                    [planet.initial_pspl_guess, planet.sfit_params, planet.revised_sfit_params,
                     planet.initial_planet_params]):

                #print(fit_type, params)

                df = {'ID': planet.lc_num}
                if params is not None:
                    #print({**df, **params})
                    df = pd.Series(data={**df, **params})
                else:
                    df = pd.Series(df)

                #print(df)

                if results[fit_type] is None:
                    results[fit_type] = df
                else:
                    results[fit_type] = pd.concat([results[fit_type], df], axis=1)

                #
        for fit_type in AllResults.fit_types:
            #print(fit_type)
            results[fit_type] = results[fit_type].transpose()
            results[fit_type].set_index('ID')
            #print(results[fit_type])

        return results

    def get_answers(self):
        all_answers = DC18Answers()
        answers = None
        for value in self.results['Initial PSPL Guess']['ID'].values:
            key = int(value)
            #print(pd.Series(data={'ID': key}))
            #print( all_answers.data.iloc[key - 1])
            df = pd.concat((pd.Series(data={'ID': key}), all_answers.data.iloc[key - 1]))

            if answers is None:
                answers = df
            else:
                answers = pd.concat((answers, df), axis=1)

        answers = answers.transpose()
        answers.set_index('ID')
        #print(answers.columns)
        return answers

    def get_ans_key(self, key):
        if key == 'rho':
            ans_key = 'rhos'
        elif '_' in key:
            ans_key = ''.join(key.split('_'))
        else:
            ans_key = key

        return ans_key

    def plot_pspl_deltas(self):
        self.plot_delta_t_0()
        self.plot_delta_u_0()
        self.plot_delta_t_E()

    def plot_planet_deltas(self):
        self.plot_delta_s()
        self.plot_delta_q()
        self.plot_delta_alpha()

    def _make_hist(self, key, frac=False, **kwargs):
        if key in ['t_0', 'u_0', 't_E']:
            fit_types = AllResults.fit_types
        else:
            fit_types = AllResults.planet_fit_types

        if 'bins' not in kwargs.keys():
            kwargs['bins'] = 20

        log_frac_keys = ['u_0', 'q', 's']

        plt.figure()
        for fit_type in fit_types:
            if frac:
                x = self.__getattribute__('delta_{0}'.format(key))[fit_type] / self.answers[self.get_ans_key(key)]
                if key in log_frac_keys:
                    x = np.log10(np.abs(x).astype(float))
            else:
                x = self.__getattribute__('delta_{0}'.format(key))[fit_type]

            plt.hist(
                x, label='{0} ({1})'.format(
                    fit_type, np.sum(pd.notna(self.__getattribute__('delta_{0}'.format(key))[fit_type]))),
                    edgecolor=self.colors[fit_type], lw=2, facecolor='none', **kwargs)

        if key == 'alpha':
            label_key = '\\alpha'
        else:
            label_key = key

        if frac:
            if key in log_frac_keys:
                plt.axvline(np.log10(0.2), color='red', label='20%')
                plt.xlabel(r'$\log (|\Delta {0} / {0}|)$'.format(label_key))
            else:
                plt.xlabel(r'$\Delta {0} / {0}$'.format(label_key))

        else:
            plt.xlabel(r'$\Delta {0}$'.format(label_key))
            plt.yscale('log')

        plt.legend()
        plt.minorticks_on()

    def plot_delta_t_0(self):
        self._make_hist('t_0', range=[-40, 40], bins=800)
        #plt.figure()
        #for i, fit_type in enumerate(AllResults.fit_types):
        #    plt.hist(
        #        self.delta_t_0[fit_type],
        #        edgecolor=AllResults.colors[i], lw=2, facecolor='none',
        #        label='{0} ({1})'.format(fit_type, np.sum(pd.notna(self.delta_t_0[fit_type]))),
        #        range=[-40, 40], bins=800)
        #
        #plt.legend()
        #plt.xlabel(r'$\Delta t_0$')
        #plt.yscale('log')
        #plt.minorticks_on()

    def plot_delta_u_0(self, frac=True):
        self._make_hist('u_0', frac=frac)
        #plt.figure()
        #for i, fit_type in enumerate(AllResults.fit_types):
        #    if frac:
        #        x = self.delta_u_0[fit_type] / self.answers['u0']
        #        x = np.log10(np.abs(x).astype(float))
        #    else:
        #        x = self.plot_delta_u_0()
        #
        #    plt.hist(
        #        x, label='{0} ({1})'.format(fit_type, np.sum(pd.notna(self.delta_u_0[fit_type]))),
        #        bins=20, zorder=-i,
        #        edgecolor=AllResults.colors[i], lw=2, facecolor='none')
        #
        #if frac:
        #    plt.axvline(-0.7, color='red')
        #    plt.xlabel(r'$\log (|\Delta u_0 / u_0|)$')
        #else:
        #    plt.xlabel(r'$\Delta u_0$')
        #    plt.yscale('log')
        #
        #plt.legend()
        #plt.minorticks_on()

    def plot_delta_t_E(self, frac=True):
        self._make_hist('t_E', frac=frac)
        #plt.figure()
        #for i, fit_type in enumerate(AllResults.fit_types):
        #    if frac:
        #        x = self.delta_t_E[fit_type] / self.answers['tE']
        #    else:
        #        x = self.plot_delta_t_E()
        #
        #    plt.hist(x, label='{0} ({1})'.format(fit_type, np.sum(pd.notna(self.delta_t_E[fit_type]))),
        #             bins=20, edgecolor=AllResults.colors[i], lw=2, facecolor='none',)
        #
        #if frac:
        #    plt.xlabel(r'$\Delta t_E / t_E$')
        #else:
        #    plt.xlabel(r'$\Delta t_E$')
        #
        #plt.legend()
        #plt.yscale('log')
        #plt.minorticks_on()

    def plot_delta_s(self, frac=True):
        self._make_hist('s', frac=frac)
        #plt.figure()
        #for i, fit_type in enumerate(AllResults.planet_fit_types):
        #    if frac:
        #        x = self.delta_s[fit_type] / self.answers['s']
        #        x = np.log10(np.abs(x).astype(float))
        #    else:
        #        x = self.plot_delta_s()
        #
        #    plt.hist(x, label='{0} ({1})'.format(fit_type, np.sum(pd.notna(self.delta_s[fit_type]))),
        #             bins=20, edgecolor=AllResults.colors[i], lw=2, facecolor='none',)
        #
        #if frac:
        #    plt.axvline(-0.7, color='red')
        #    plt.xlabel(r'$\log |\Delta s / s|$')
        #else:
        #    plt.xlabel(r'$\Delta s$')
        #
        #plt.legend()
        #plt.yscale('log')
        #plt.minorticks_on()

    def plot_delta_q(self, frac=True):
        self._make_hist('q', frac=frac)
        #plt.figure()
        #for i, fit_type in enumerate(AllResults.planet_fit_types):
        #    if frac:
        #        x = self.delta_q[fit_type] / self.answers['q']
        #        x = np.log10(np.abs(x).astype(float))
        #    else:
        #        x = self.plot_delta_q()
        #
        #    plt.hist(x, label='{0} ({1})'.format(fit_type, np.sum(pd.notna(self.delta_q[fit_type]))),
        #             bins=20, edgecolor=AllResults.colors[i], lw=2, facecolor='none',)
        #
        #if frac:
        #    plt.axvline(-0.7, color='red')
        #    plt.xlabel(r'$\log |\Delta q / q|$')
        #else:
        #    plt.xlabel(r'$\Delta q$')
        #
        #plt.legend()
        #plt.yscale('log')
        #plt.minorticks_on()

    def plot_delta_alpha(self, frac=False):
        self._make_hist('alpha', frac=frac)
        #plt.figure()
        #for i, fit_type in enumerate(AllResults.planet_fit_types):
        #    if frac:
        #        x = self.delta_alpha[fit_type] / self.answers['alpha']
        #    else:
        #        x = self.delta_alpha[fit_type]
        #
        #    plt.hist(x, label='{0} ({1})'.format(fit_type, np.sum(pd.notna(self.delta_alpha[fit_type]))),
        #             bins=20, edgecolor=AllResults.colors[i], lw=2, facecolor='none',)
        #
        #if frac:
        #    plt.xlabel(r'$\Delta \alpha / \alpha$')
        #else:
        #    plt.xlabel(r'$\Delta \alpha$')
        #
        #plt.legend()
        #plt.yscale('log')
        #plt.minorticks_on()

    def print_median_deltas(self):
        for fit_type in AllResults.fit_types:
            print(fit_type)
            headstr = '{0:6}'.format('')
            medstr = '{0:6}'.format('med')
            fracstr = '{0:6}'.format('frac')
            for key in self.results[fit_type].columns:
                if key != 'ID':
                    headstr += '{0:>9} '.format('d({0})'.format(key))
                    x = self.__getattribute__('delta_{0}'.format(key))[fit_type]
                    value = self.answers[self.get_ans_key(key)]
                    medstr += '{0:9.4f} '.format(np.nanmedian(x.astype(float)))
                    fracstr += '{0:9.4f} '.format(np.nanmedian(x.astype(float) / value.astype(float)))

            print(headstr)
            print(medstr)
            print(fracstr)

    @property
    def delta_t_0(self):
        if self._delta_t_0 is None:
            delta_t_0 = {}
            for fit_type in AllResults.fit_types:
                delta_t_0[fit_type] = self.answers['t0'] - self.results[fit_type]['t_0'] + 2458234.

            self._delta_t_0 = delta_t_0

        return self._delta_t_0

    @property
    def delta_u_0(self):
        if self._delta_u_0 is None:
            delta_u_0 = {}
            for fit_type in AllResults.fit_types:
                delta_u_0[fit_type] = self.answers['u0'] - self.results[fit_type]['u_0']

            self._delta_u_0 = delta_u_0

        return self._delta_u_0

    @property
    def delta_t_E(self):
        if self._delta_t_E is None:
            delta_t_E = {}
            for fit_type in AllResults.fit_types:
                delta_t_E[fit_type] = self.answers['tE'] - self.results[fit_type]['t_E']

            self._delta_t_E = delta_t_E

        return self._delta_t_E

    @property
    def delta_rho(self):
        if self._delta_rho is None:
            delta_rho = {}
            for fit_type in AllResults.planet_fit_types:
                delta_rho[fit_type] = self.answers['rhos'] - self.results[fit_type]['rho']

            self._delta_rho = delta_rho

        return self._delta_t_E

    @property
    def delta_s(self):
        if self._delta_s is None:
            delta_s = {}
            for fit_type in AllResults.planet_fit_types:
                delta_s[fit_type] = self.answers['s'] - self.results[fit_type]['s']

            self._delta_s = delta_s

        return self._delta_s

    @property
    def delta_q(self):
        if self._delta_q is None:
            delta_q = {}
            for fit_type in AllResults.planet_fit_types:
                delta_q[fit_type] = self.answers['q'] - self.results[fit_type]['q']

            self._delta_q = delta_q

        return self._delta_q

    @property
    def delta_alpha(self):
        if self._delta_alpha is None:
            delta_alpha = {}
            for fit_type in AllResults.planet_fit_types:
                delta_alpha[fit_type] = self.answers['alpha'] - self.results[fit_type]['alpha']

            self._delta_alpha = delta_alpha

        return self._delta_alpha


if __name__ == '__main__':
    results = AllResults(path=os.path.join('temp_output'))
    results.plot_pspl_deltas()
    results.plot_planet_deltas()
    results.print_median_deltas()
    plt.show()
