#include "snpsamplinge.hh"
#include "log.hh"
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_blas.h>

SNPSamplingE::SNPSamplingE(Env &env, SNP &snp)
  :_env(env), _snp(snp),
   _n(env.n), _k(env.k), _l(_env.l),
   _t(env.t), _nthreads(_env.nthreads),
   _iter(0), _alpha(_k), _loc(0),
   _eta(_k,_t),
   _gamma(_n,_k), 
   _lambda(_l,_k,_t),
   _lambdat(_k,_t),
   _tau0(env.tau0 + 1), _kappa(env.kappa),
   _nodetau0(env.nodetau0 + 1), _nodekappa(env.nodekappa),
   _rho_indiv(_n),
   _c_indiv(_n),
   _nodeupdatec(_n),
   _start_time(time(0)),
   _Elogtheta(_n,_k),
   _Elogbeta(_l,_k,_t),
   _Etheta(_n,_k),
   _Ebeta(_l,_k),
   _shuffled_nodes(_n),
   _max_t(-2147483647),
   _max_h(-2147483647),
   _prev_h(-2147483647),
   _prev_w(-2147483647),
   _prev_t(-2147483647),
   _nh(0), _nt(0),
   _sampled_loc(0),
   _total_locations(0),
   _hol_mode(false),
   _phidad(_n,_k), _phimom(_n,_k),
   _phinext(_k), _lambdaold(_k,_t),
   _v(_k,_t),
   _pi(_n*2),
   _trait(_n),
   _diff_dev(_l),
   _run_gcat(_env.run_gcat)
{
  printf("+ initialization begin\n");
  fflush(stdout);

  _total_locations = _n * _l;

  info("+ running inference on %lu nodes\n", _n);

  _alpha.set_elements(env.alpha);
  info("alpha set to %s\n", _alpha.s().c_str());

  double **d = _eta.data();
  for (uint32_t i = 0; i < _eta.m(); ++i) {
    d[i][0] = 1.0;
    d[i][1] = 1.0;
  }

  // random number generation
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  if (env.seed)
    gsl_rng_set(_r, _env.seed);


  unlink(Env::file_str("/likelihood-analysis.txt").c_str());

  _vf = fopen(Env::file_str("/validation.txt").c_str(), "w");
  if (!_vf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }

  if (_env.compute_beta) {
    _env.online_iterations = 100; // tightly optimize given the thetas

    init_heldout_sets();
    if (_nthreads > 0) {
      Thread::static_initialize();
      PhiRunnerE::static_initialize();
      start_threads();
    }
    lerr("done starting threads");
    
    load_gamma();
    estimate_all_theta();
    lerr("done estimating all theta");
    if (_env.locations_file == "") {
      compute_all_lambda();
      estimate_all_beta();
      save_beta();
    } else
      compute_and_save_beta();
    exit(0);
  }

  init_heldout_sets();
  init_gamma();
  init_lambda();

  estimate_all_theta();

  printf("+ computing initial heldout likelihood\n");
  compute_likelihood(true, true);
  if (_env.use_test_set)
    compute_likelihood(true, false);
  save_gamma();
  printf("\n+ computing initial training likelihood\n");
  printf("+ done..\n");

  gettimeofday(&_last_iter, NULL);
  printf("+ initialization end\n");
  fflush(stdout);

  if (_nthreads > 0) {
    Thread::static_initialize();
    PhiRunnerE::static_initialize();
    start_threads();
  }
}

SNPSamplingE::~SNPSamplingE()
{
  fclose(_vf);
  fclose(_tf);
  fclose(_lf);
  fclose(_tef);
  fclose(_vef);
}

void
SNPSamplingE::init_heldout_sets()
{
  if (_env.use_test_set)
    set_test_sample();
  set_validation_sample();

}

void
SNPSamplingE::set_test_sample()
{
  uint32_t per_loc_h = _n * _env.test_ratio * 100 / 5;
  uint32_t nlocs = _l * _env.test_ratio;
  map<uint32_t, bool> lm;
  do {
    uint32_t loc = gsl_rng_uniform_int(_r, _l);
    map<uint32_t, bool>::const_iterator z = lm.find(loc);
    if (z != lm.end()) 
      continue;
    else
      lm[loc] = true;
    
    uint32_t c = 0;
    while (c < per_loc_h) {
      uint32_t indiv = gsl_rng_uniform_int(_r, _n);
      if (kv_ok(indiv, loc)) {
	KV kv(indiv, loc);
	_test_map[kv] = true;
	c++;
      }
    }
  } while (lm.size() < nlocs);
}

void
SNPSamplingE::set_validation_sample2()
{
  for (uint32_t l = 0; l < _l; ++l) {
    // for each location keep aside h individuals
    uint32_t h = _env.heldout_indiv_ratio * _n;
    if (h < 1)
      h = 1;
    else if (h > 10)
      h = 10;
    
    uint32_t c = 0;
    do {
      uint32_t indiv = gsl_rng_uniform_int(_r, _n);
      if (kv_ok(indiv, l)) {
	KV kv(indiv, l);
	_validation_map[kv] = true;
	c++;
      }
    } while (c < h);
  }
}

void
SNPSamplingE::set_validation_sample()
{
  uint32_t per_loc_h = _n < 2000 ? (_n / 10) : (_n / 100);
  uint32_t nlocs = _l * _env.validation_ratio;
  map<uint32_t, bool> lm;
  do {
    uint32_t loc = gsl_rng_uniform_int(_r, _l);
    map<uint32_t, bool>::const_iterator z = lm.find(loc);
    if (z != lm.end()) 
      continue;
    else
      lm[loc] = true;
    
    uint32_t c = 0;
    while (c < per_loc_h) {
      uint32_t indiv = gsl_rng_uniform_int(_r, _n);
      if (kv_ok(indiv, loc)) {
	KV kv(indiv, loc);
	_validation_map[kv] = true;
	c++;
      }
    }
  } while (lm.size() < nlocs);
}

void
SNPSamplingE::init_gamma()
{
  double **d = _gamma.data();
  for (uint32_t i = 0; i < _n; ++i) {
    for (uint32_t j = 0; j < _k; ++j)  {
      double v = (_k < 100) ? 1.0 : (double)100.0 / _k;
      d[i][j] = gsl_ran_gamma(_r, 100 * v, 0.01);
    }
  }
  PopLib::set_dir_exp(_gamma, _Elogtheta);
}

void
SNPSamplingE::init_lambda()
{
  double ***ld = _lambda.data();
  const double **etad = _eta.const_data();
  for (uint32_t l = 0; l < _l; ++l)
    for (uint32_t k = 0; k < _k; ++k)
      for (uint32_t t = 0; t < _t; ++t) {
	ld[l][k][t] = etad[k][t];
      }
  PopLib::set_dir_exp(_lambda, _Elogbeta);
}

int
SNPSamplingE::start_threads()
{
  for (uint32_t i = 0; i < _nthreads; ++i) {
    PhiRunnerE *t = new PhiRunnerE(_env, &_r, 
				   _iter, _x, _n, _k, 
				   0, _t, _snp, *this,
				   _out_q, _in_q, _cm);
    if (t->create() < 0)
      return -1;
    _thread_map[t->id()] = t;
  }
  return 0;
}

void
SNPSamplingE::update_lambda(uint32_t loc)
{
  const yval_t ** const snpd = _snp.y().const_data();
  double **ld = _lambda.data()[loc];
  double **ldt = _lambdat.data();
  for (uint32_t k = 0; k < _k; ++k) {
    ld[k][0] = _env.eta0 + ldt[k][0];
    ld[k][1] = _env.eta1 + ldt[k][1];
  }
}

void
SNPSamplingE::estimate_beta(uint32_t loc)
{
  const double ***ld = _lambda.const_data();
  double **betad = _Ebeta.data();
  double ***elogbeta = _Elogbeta.data();

  for (uint32_t k = 0; k < _k; ++k) {
    double s = .0;
    for (uint32_t t = 0; t < _t; ++t)
      s += ld[loc][k][t];
    betad[loc][k] = ld[loc][k][0] / s;
    
    double psi_sum = gsl_sf_psi(s);
    elogbeta[loc][k][0] = gsl_sf_psi(ld[loc][k][0]) - psi_sum;
    elogbeta[loc][k][1] = gsl_sf_psi(ld[loc][k][1]) - psi_sum;
  }
}

void
SNPSamplingE::split_all_indivs()
{
  // split indivs into _nthread chunks
  uint32_t chunk_size = (int)(((double)_n) / _nthreads);
  uint32_t t = 0, c = 0;
  for (uint32_t i = 0; i < _n; ++i) {
    ChunkMap::iterator it = _chunk_map.find(t);
    if (it == _chunk_map.end()) {
      IndivsList *il = new IndivsList;
      _chunk_map[t] = il;
    }
    IndivsList *il = _chunk_map[t];
    il->push_back(i);
    c++;
    if (c >= chunk_size && t < (uint32_t)_nthreads - 1) {
      c = 0;
      t++;
    }
  }
}

void
SNPSamplingE::optimize_lambda(uint32_t loc)
{
  _x = 0;
  do {
    debug("x = %d", x);
    for (ChunkMap::iterator it = _chunk_map.begin(); 
	 it != _chunk_map.end(); ++it) {
      IndivsList *il = it->second;
      debug("pushing chunk of size %d", il->size());
      _out_q.push(il);
    }

    _cm.lock();
    _cm.broadcast();
    _cm.unlock();
    
    _lambdat.zero();
    uint32_t nt = 0;
    do {
      // do not delete p!
      pthread_t *p = _in_q.pop();
      assert(p);
      PhiRunnerE *t = _thread_map[*p];
      debug("main: threads %d done (id:%ld)", nt+1, t->id());
      const Matrix &lambdat = t->lambdat();
      const double **ldt_t = lambdat.const_data();
      double **ldt = _lambdat.data();
      for (uint32_t k = 0; k < _k; ++k)
	for (uint32_t r = 0; r < _t; ++r)
	  ldt[k][r] += ldt_t[k][r];
      nt++;
    } while (nt != _nthreads || !_in_q.empty());
    
    assert (nt == _nthreads);

    _lambdaold.copy_from(loc, _lambda);
    update_lambda(loc);
    estimate_beta(loc);
    sub(loc, _lambda, _lambdaold, _v);

    _x++;
    
    if (_v.abs_mean() < _env.meanchangethresh)
      break;
  } while (_x < _env.online_iterations);
}

void
SNPSamplingE::compute_all_lambda()
{
  split_all_indivs();
  for (uint32_t loc = 0; loc < _l; ++loc) {
    _loc = loc;
    optimize_lambda(loc);
    _iter++;
    if (_loc % 100 == 0) {
      printf("\rloc = %d took %d secs", _iter, duration());
      fflush(stdout);
    }
  }
}


void
SNPSamplingE::compute_and_save_beta()
{
  lerr("within compute_and_save_beta()");
  FILE *f = fopen(_env.locations_file.c_str(), "r");
  vector<uint32_t> locs;
  uint32_t loc;
  char b[4096*4];
  while (!feof(f)) {
    if (fscanf(f, "%d\t%*[^\n]s\n", &loc, b) >= 0) {
      lerr("loc = %d", loc);
      locs.push_back(loc);
    }
  }
  fclose(f);
  lerr("locs size = %d", locs.size());
  
  split_all_indivs();
  for (uint32_t i = 0; i < locs.size(); ++i) {
    uint32_t loc = locs[i];
    _loc = loc;
    optimize_lambda(loc);
    _iter++;
    if (_loc % 100 == 0) {
      printf("\rloc = %d took %d secs", _iter, duration());
      fflush(stdout);
    }
  }
  save_beta(locs);
}



void
SNPSamplingE::infer()
{
  split_all_indivs();

  while (1) {
    _loc = gsl_rng_uniform_int(_r, _l);
    debug("LOC = %d", _loc);
    optimize_lambda(_loc);
    
    // threads update gamma in the next iteration
    // prior to updating phis

    debug("x = %d, lambda = %s", _x, _lambda.s(_loc).c_str());
    debug("loc = %d, beta = %s\n", _loc, _Ebeta.s(_loc).c_str());
    debug("n  30, gamma = %s", _gamma.s(30).c_str());

    _iter++;

    if (_iter % 100 == 0) {
      printf("\riteration = %d took %d secs", _iter, duration());
      fflush(stdout);
    }

    if (_iter % _env.reportfreq == 0) {
      printf("iteration = %d took %d secs\n", 
	     _iter, duration());
      lerr("iteration = %d took %d secs\n", _iter, duration());
      lerr("computing heldout likelihood @ %d secs", duration());
      if(compute_likelihood(false, true) == -1)
        break;
      if (_env.use_test_set)
	if(compute_likelihood(false, false) == -1)
          break;
      lerr("saving theta @ %d secs", duration());
      save_model();
      lerr("done @ %d secs", duration());
    }

    if (_env.terminate) {
      save_model();
      if(_run_gcat)
        break;
      else
        exit(0);
    }
  }
  if(_run_gcat) {
    printf("\nDone with inference. Now running GCAT.\n");
    gcat();
    printf("\nDone with GCAT. Now saving difference in deviance.\n");
    save_diff_dev();
    printf("\nDone!\n");
  }
}

double
SNPSamplingE::compute_likelihood(bool first, bool validation)
{
  _hol_mode = true;
  uint32_t k = 0;
  double s = .0;

  SNPMap *mp = NULL;
  FILE *ff = NULL;
  if (validation) {
    mp = &_validation_map;
    ff = _vf;
  } else {
    mp = &_test_map;
    ff = _tf;
  }

  SNPByLoc m;
  for (SNPMap::const_iterator i = mp->begin(); i != mp->end(); ++i) {
    const KV &kv = i->first;

    uint32_t indiv = kv.first;
    uint32_t loc = kv.second;

    vector<uint32_t> &v = m[loc];
    v.push_back(indiv);
  }

  vector<uint32_t> indivs;
  uint32_t sz = 0;
  for (SNPByLoc::const_iterator i = m.begin(); i != m.end(); ++i) {
    uint32_t loc = i->first;
    indivs = i->second;
    printf("\rdone:%.2f%%", ((double)sz / m.size())*100);
    double u = snp_likelihood(loc, indivs, first);
    s += u;
    k += indivs.size();
    sz++;
  }
  fprintf(ff, "%d\t%d\t%.9f\t%d\t%f\n", _iter, duration(), (s / k), k, exp(s/k));
  fflush(ff);
  
  double a = (s / k);

  if (!validation) {
    _hol_mode = false;
    return 0;
  }
  
  bool stop = false;
  int why = -1;
  if (_iter > 2000) {
    if (a > _prev_h && 
	_prev_h != 0 && fabs((a - _prev_h) / _prev_h) < _env.stop_threshold) {
      stop = true;
      why = 0;
    } else if (a < _prev_h)
      _nh++;
    else if (a > _prev_h)
      _nh = 0;

    if (a > _max_h)
      _max_h = a;
    
    if (_nh > 3) {
      why = 1;
      stop = true;
    }
  }
  _prev_h = a;

  if (stop) {
    double v = 0; //validation_likelihood();
    double t = 0; //t = training_likelihood();

    if (_env.use_validation_stop) {
      _hol_mode = false;
      save_model();
      if(_run_gcat)
        return -1;
      else
        exit(0);
    }
  }
  _hol_mode = false;
  return (s / k) / _n;
}

void
SNPSamplingE::save_gamma()
{
  FILE *f = fopen(add_iter_suffix("/gamma").c_str(), "w");
  FILE *g = fopen(add_iter_suffix("/theta").c_str(), "w");
  if (!f || !g)  {
    lerr("cannot open gamma/theta file:%s\n",  strerror(errno));
    exit(-1);
  }
  double **gd = _gamma.data();
  double **td = _Etheta.data();
  for (uint32_t n = 0; n < _n; ++n) {
    string s = _snp.label(n);
    if (s == "")
      s = "unknown";
    double max = .0;
    uint32_t max_k = 0;
    for (uint32_t k = 0; k < _k; ++k) {
      fprintf(f, "%.8f\t", gd[n][k]);
      fprintf(g, "%.8f\t", td[n][k]);
      if (gd[n][k] > max) {
        max = gd[n][k];
        max_k = k;
      }
    }
    fprintf(f,"\n", max_k);
    fprintf(g,"\n", max_k);
  }
  fclose(f);
  fclose(g);
}

string
SNPSamplingE::add_iter_suffix(const char *c)
{
  ostringstream sa;
  if (_env.file_suffix)
    sa << c << "_" << _iter << ".txt";
  else
    sa << c << ".txt";
  return Env::file_str(sa.str());
}

void
SNPSamplingE::save_model()
{
  save_gamma();
}

void
SNPSamplingE::estimate_all_theta()
{
  const double ** const gd = _gamma.const_data();
  double **theta = _Etheta.data();
  for (uint32_t n = 0; n < _n; ++n) {
    double s = .0;
    for (uint32_t k = 0; k < _k; ++k)
      s += gd[n][k];
    assert(s);
    for (uint32_t k = 0; k < _k; ++k)
      theta[n][k] = gd[n][k] / s;
  } 
  PopLib::set_dir_exp(_gamma, _Elogtheta);
}

void
SNPSamplingE::estimate_all_beta()
{
  const double ***ld = _lambda.const_data();
  double **betad = _Ebeta.data();

  for (uint32_t loc = 0; loc < _l; ++loc) {
    for (uint32_t k = 0; k < _k; ++k) {
      double s = .0;
      for (uint32_t t = 0; t < _t; ++t)
	s += ld[loc][k][t];
      betad[loc][k] = ld[loc][k][0] / s;
    }
  }
}

inline void
SNPSamplingE::update_phimom(uint32_t n, uint32_t loc)
{
  const double ** const elogthetad = _Elogtheta.const_data();
  const double ** const elogbetad = _Elogbeta.const_data()[loc];
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogbetad[k][0];
  _phinext.lognormalize();
  _phimom.set_elements(n, _phinext);
}

inline void
SNPSamplingE::update_phidad(uint32_t n, uint32_t loc)
{
  const double ** const elogthetad = _Elogtheta.const_data();
  const double ** const elogbetad = _Elogbeta.const_data()[loc];
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogbetad[k][1];
  _phinext.lognormalize();
  _phidad.set_elements(n, _phinext);
}

int
PhiRunnerE::do_work()
{
  bool first = true;
  _idptr = new pthread_t(pthread_self());
  _oldilist = NULL;
  
  do {
    IndivsList *ilist = _out_q.pop();
    debug("thread = %ld, popped size %d, at0: %d\n", 
	 id(), ilist->size(), (*ilist)[0]);
    if (first || _prev_iter != _iter) {
      debug("thread = %ld, NEW loc = %d\n", id(), _pop.sampled_loc());
      
      if (!first) {
	if (!_prev_hol_mode) {
	  update_gamma();
	  estimate_theta();
	}
      }
      reset(_pop.sampled_loc());
      first = false;
    }
    
    _oldilist = ilist;
    _lambdat.zero();
    process(*ilist);

    _in_q.push(_idptr);
    
    _cm.lock();
    while (_x == _prev_x && _iter == _prev_iter)
      _cm.wait();
    _prev_x = _x;
    _cm.unlock();

  } while (1);
}

void
SNPSamplingE::update_rho_indiv(uint32_t n)
{
  _rho_indiv[n] = pow(_nodetau0 + _c_indiv[n], -1 * _nodekappa);
  _c_indiv[n]++;
}

void
PhiRunnerE::update_gamma(const IndivsList &indivs)
{
  const double **phidadd = _phidad.const_data();
  const double **phimomd = _phimom.const_data();
  const yval_t ** const snpd = _snp.y().const_data();

  double gamma_scale = _env.l;
  double **gd = _pop.gamma().data();

  // no locking needed
  // each thread owns it's own set of indivs
  for (uint32_t i = 0; i < indivs.size(); ++i) {
    uint32_t n = indivs[i];
    if (!_pop.kv_ok(n, _loc))
      continue;

    _pop.update_rho_indiv(n);
    yval_t y = snpd[n][_loc];
    for (uint32_t k = 0; k < _k; ++k) {
      gd[n][k] += _pop.rho_indiv(n) *					\
	(_pop.alpha(k) + (gamma_scale * (y * phimomd[n][k] + (2 - y) * phidadd[n][k])) - gd[n][k]);
    }
  }
}

void
PhiRunnerE::estimate_theta(const IndivsList &indivs)
{
  const double ** const gd = _pop.gamma().const_data();
  double **theta = _pop.Etheta().data();
  double **elogtheta = _pop.Elogtheta().data();
  
  for (uint32_t i = 0; i < indivs.size(); ++i)  {
    uint32_t n = indivs[i];
    double s = .0;
    for (uint32_t k = 0; k < _k; ++k)
      s += gd[n][k];
    assert(s);
    double psi_sum = gsl_sf_psi(s);
    for (uint32_t k = 0; k < _k; ++k) {
      theta[n][k] = gd[n][k] / s;
      elogtheta[n][k] = gsl_sf_psi(gd[n][k]) - psi_sum;
    }
  }
}

void
PhiRunnerE::update_lambda_t(const IndivsList &indivs)
{
  const double **phidadd = _phidad.const_data();
  const double **phimomd = _phimom.const_data();
  const yval_t ** const snpd = _snp.y().const_data();

  double **ldt = _lambdat.data();
  for (uint32_t k = 0; k < _k; ++k) {
    for (uint32_t i = 0; i < indivs.size(); ++i)  {
      uint32_t n = indivs[i];
      if (!_pop.kv_ok(n, _loc))
	continue;
      ldt[k][0] += phimomd[n][k] * snpd[n][_loc];
      ldt[k][1] += phidadd[n][k] * (2 - snpd[n][_loc]);
    }
  }
}

void
SNPSamplingE::save_beta()
{
  const double **ebeta = _Ebeta.const_data();
  FILE *f = fopen(add_iter_suffix("/beta").c_str(), "w");
  if (!f)  {
    lerr("cannot open beta or lambda file:%s\n",  strerror(errno));
    exit(-1);
  }
  for (uint32_t l = 0; l < _l; ++l) {
    fprintf(f, "%d\t", l);
    for (uint32_t k = 0; k < _k; ++k) {
      fprintf(f, "%.8f\t", ebeta[l][k]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
}

void
SNPSamplingE::save_beta(const vector<uint32_t> &locs)
{
  const double **ebeta = _Ebeta.const_data();
  FILE *f = fopen(add_iter_suffix("/beta").c_str(), "w");
  if (!f)  {
    lerr("cannot open beta or lambda file:%s\n",  strerror(errno));
    exit(-1);
  }
  for (uint32_t i = 0; i < locs.size(); ++i) {
    uint32_t loc = locs[i];
    fprintf(f, "%d\t", loc);
    for (uint32_t k = 0; k < _k; ++k) {
      fprintf(f, "%.8f\t", ebeta[loc][k]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
}

void
SNPSamplingE::load_gamma()
{
  double **gammad = _gamma.data();
  FILE *gammaf = fopen("gamma.txt", "r");
  if (!gammaf)  {
    lerr("cannot open gamma file:%s\n",  strerror(errno));
    exit(-1);
  }

  int sz = 128 * _k;
  uint32_t n = 0;
  char *line = (char *)malloc(sz);
  while (!feof(gammaf)) {
    if (fgets(line, sz, gammaf) == NULL) 
      break;
    
    uint32_t k = 0;
    char *p = line;
    do {
      char *q = NULL;
      double d = strtod(p, &q);
      if (p == q) {
        if (k < _k - 1) {
          fprintf(stderr, "error parsing gamma file\n");
          assert(0);
          }
          break;
      }
      p = q;
      gammad[n][k] = d;
      k++;
    } while (p != NULL);
    n++;
    memset(line, 0, sz);
  }
  assert (n = _n);
  fclose(gammaf);

  FILE *f = fopen(Env::file_str("/gammasave.txt").c_str(), "w");
  if (!f)  {
    lerr("cannot open gammasave file:%s\n",  strerror(errno));
    exit(-1);
  }
  double **gd = _gamma.data();
  for (uint32_t n = 0; n < _n; ++n) {
    string s = _snp.label(n);
    if (s == "")
      s = "unknown";
    fprintf(f, "%d\t%s\t", n, s.c_str());
    double max = .0;
    uint32_t max_k = 0;
    for (uint32_t k = 0; k < _k; ++k) {
      fprintf(f, "%.8f\t", gd[n][k]);
      if (gd[n][k] > max) {
        max = gd[n][k];
        max_k = k;
      }
    }
    fprintf(f,"%d\n", max_k);
  }
  fclose(f);
}

// for GCAT
void
SNPSamplingE::split_all_SNPs()
{
    // split SNPs into _nthread chunks
    uint32_t chunk_size = (int)(((double)_l) / _nthreads);
    uint32_t t = 0, c = 0;
    
    for (uint32_t i = 0; i < _l; ++i) {
        SNPChunkMap::iterator it = _snp_chunk_map.find(t);
        if (it == _snp_chunk_map.end()) {
            SNPsList *il = new SNPsList;
            _snp_chunk_map[t] = il;
        }
        SNPsList *il = _snp_chunk_map[t];
        _snp_chunk_map[t] = il;
        il->push_back(i);
        c++;
        if (c >= chunk_size && t < (uint32_t)_nthreads - 1) {
            c = 0;
            t++;
        }
    }
}

int
SNPSamplingE::read_trait(string s)
{
  double *trait_d = _trait.data();

  FILE *f = fopen(s.c_str(), "r");
  if (!f) {
    lerr("cannot open file %s:%s", s.c_str(), strerror(errno));
    return -1;
  }
  
  // Assuming FAM file format
  char tmpbuf[2048*10];
  char *token;

  for (uint32_t i = 0; i < _n; ++i) {
    for(uint32_t j = 0; j < 6; ++j) {
      if (fscanf(f, "%s\n", tmpbuf) < 0) {
        printf("Error: unexpected lines in trait file\n");
        exit(-1);
      }
    }
    trait_d[i] = strtod(tmpbuf, NULL);
  }
  fflush(stdout);
  fclose(f);

  return 0;
}

// TODO: make multithreaded!
void
SNPSamplingE::gcat()
{
  // Calculate associations over all SNPs
  double *diff_dev_d = _diff_dev.data();

  // Initialize vars for IRLS 
  _y_dbl = gsl_vector_alloc(_n*2); // Doubled genotypes
  _p = gsl_vector_alloc(_n*2); // MLE
  
  int cols_null = 1;
  _X_null = gsl_matrix_alloc(_n*2, cols_null); // Intercept
  _b_null = gsl_vector_alloc(cols_null); // Beta for null
  _bl_null = gsl_vector_alloc(cols_null); // Beta last for null
  _f_null = gsl_vector_alloc(cols_null);
  _W_null = gsl_matrix_alloc(cols_null, cols_null);
  _Wo_null = gsl_matrix_alloc(cols_null, cols_null); // Inverted W
  _W_permut_null = gsl_permutation_alloc(cols_null);

  int cols_alt = cols_null+1;
  _X_alt = gsl_matrix_alloc(_n*2, cols_alt); // Intercept and trait
  _b_alt = gsl_vector_alloc(cols_alt); // Beta for alt
  _bl_alt = gsl_vector_alloc(cols_alt); // Beta last for alt
  _f_alt = gsl_vector_alloc(cols_alt);
  _W_alt = gsl_matrix_alloc(cols_alt, cols_alt);
  _Wo_alt = gsl_matrix_alloc(cols_alt, cols_alt); // Inverted W
  _W_permut_alt = gsl_permutation_alloc(cols_alt);

  const double ***ld = _lambda.const_data(); // Used to calculate pi

  // Calculate thetas
  const double ** const gd = _gamma.const_data();
  double **theta = _Etheta.data();
  for (uint32_t n = 0; n < _n; ++n) {
      double s = .0;
      for (uint32_t k = 0; k < _k; ++k)
          s += gd[n][k];
      assert(s);
      for (uint32_t k = 0; k < _k; ++k)
          theta[n][k] = gd[n][k] / s;
  }

  // Calculate population struct est. at each SNP, run assoc test
  for (uint32_t loc = 0; loc < _l; ++loc) {
    for(uint32_t n = 0; n < _n; ++n) {
      _pi[n] = 0;
      _pi[n+_n] = 0;
    }
    for (uint32_t k = 0; k < _k; ++k) {
        double s = .0;
        for (uint32_t t = 0; t < _t; ++t)
            s += ld[loc][k][t];
        double beta = ld[loc][k][0] / s;
        for(uint32_t n = 0; n < _n; ++n) {
            _pi[n] += beta*theta[n][k];
            _pi[n+_n] += beta*theta[n][k];
        }
    }
    diff_dev_d[loc] = calc_diff_dev(loc);
  }

  // Clean up
  gsl_vector_free(_y_dbl);
  gsl_vector_free(_p);

  gsl_matrix_free(_X_null); 
  gsl_vector_free(_b_null);
  gsl_vector_free(_bl_null);
  gsl_vector_free(_f_null);
  gsl_matrix_free(_W_null);
  gsl_matrix_free(_Wo_null);
  gsl_permutation_free(_W_permut_null);

  gsl_matrix_free(_X_alt); 
  gsl_vector_free(_b_alt);
  gsl_vector_free(_bl_alt);
  gsl_vector_free(_f_alt);
  gsl_matrix_free(_W_alt);
  gsl_matrix_free(_Wo_alt);
  gsl_permutation_free(_W_permut_alt);
}

void
SNPSamplingE::save_diff_dev()
{
  FILE *f = fopen(add_iter_suffix("/diffDev").c_str(), "w");
  if (!f)  {
    lerr("cannot open diffDev file:%s\n",  strerror(errno));
    exit(-1);
  }
  for (uint32_t loc = 0; loc < _l; ++loc)
    fprintf(f, "%.8f\n", _diff_dev[loc]);
  fclose(f);
}

// TODO: allow for user-inputted covariates
// Runs GCAT on one SNP location
double
SNPSamplingE::calc_diff_dev(uint32_t loc) {
  // Iteration values
  int i, j;

  const yval_t ** const snpd = _snp.y().const_data();
  
  // Set values of X
  for(int n = 0; n < _n; n++) {
    // Set values of X
    gsl_matrix_set(_X_null, n, 0, 1);
    gsl_matrix_set(_X_null, n+_n, 0, 1);
    gsl_matrix_set(_X_alt, n, 0, 1);
    gsl_matrix_set(_X_alt, n+_n, 0, 1);
    gsl_matrix_set(_X_alt, n, 1, _trait[n]);
    gsl_matrix_set(_X_alt, n+_n, 1, _trait[n]);
    // Double genotype and subtract population structure offset
    if (snpd[n][loc] == 2) {
      gsl_vector_set(_y_dbl, n, 1.0);
      gsl_vector_set(_y_dbl, n+_n, 1.0);
    } else if(snpd[n][loc] == 1) {
      gsl_vector_set(_y_dbl, n, 1.0);
      gsl_vector_set(_y_dbl, n+_n, 0.0);
    } else if(snpd[n][loc] == 0) {
      gsl_vector_set(_y_dbl, n, 0.0);
      gsl_vector_set(_y_dbl, n+_n, 0.0);
    }
  }

  gsl_vector_set_zero(_b_null);
  gsl_vector_set_zero(_bl_null);

  // Algorithmic constants
  _max_iter_irls = 10;
  _tol_irls = 1e-6;
  // Calculate deviance for null model
  double dev_null = calc_dev(true);

  // Set b_alt
  gsl_vector_set(_b_alt, 0, gsl_vector_get(_b_null, 0));
  gsl_vector_set(_bl_alt, 0, gsl_vector_get(_bl_null, 0));
  gsl_vector_set(_b_alt, 1, 0);
  gsl_vector_set(_bl_alt, 1, 0);

  // Calculate deviance for alt model
  double dev_alt = calc_dev(false);

  double diff_dev = -2*(dev_null - dev_alt);
  return diff_dev;
}

// Calculates deviance of a model using IRLS (iteratively-reweighted least squares)
double
SNPSamplingE::calc_dev(bool null_model) {
  // Stopping condition
  double max_rel_change;
  double rel_change;
  // Values for iteration
  int signum;
  long i, j, k;
  int n_iter;
  double y_i;
  double bl_i;
  double p_i, p_k;
  double w_ij;

  gsl_matrix *X;
  gsl_vector *b;
  gsl_vector *bl;
  gsl_vector *f;
  gsl_matrix *W;
  gsl_matrix *Wo; // Inverted W
  gsl_permutation *W_permut;
  if (null_model) {
    X = _X_null;
    b = _b_null;
    bl = _bl_null;
    f = _f_null;
    W = _W_null;
    Wo = _Wo_null;
    W_permut = _W_permut_null;
  } else {
    X = _X_alt;
    b = _b_alt;
    bl = _bl_alt;
    f = _f_alt;
    W = _W_alt;
    Wo = _Wo_alt;
    W_permut = _W_permut_alt;
  }
  // Utility sizes
  long X_rows = X->size1;
  long X_cols = X->size2;

  // TODO: SET ALL VECTS ETC. TO 0S
  gsl_vector_set_zero(_p);
  gsl_vector_set_zero(f);
  gsl_matrix_set_zero(Wo);
  gsl_permutation_init(W_permut);
  
  for(n_iter = 0; n_iter < _max_iter_irls; n_iter++) {
    // p <- as.vector(1/(1 + exp(-X %*% b)))
    gsl_blas_dgemv(CblasNoTrans, -1.0, X, b, 0.0, _p);
    for(i = 0; i < X_rows; i++) {
      p_i = gsl_vector_get(_p, i) - _pi[i];
      gsl_vector_set(_p, i, 1/(1+exp(gsl_vector_get(_p, i))));
    }
    // var.b <- solve(crossprod(X, p * (1 - p) * X))
    gsl_matrix_set_zero(W);
    for(i = 0; i < X_cols; i++) {
      for(j = i; j < X_cols; j++) {
        for(k = 0; k < X_rows; k++) {
          p_k = gsl_vector_get(_p, k);
          w_ij = gsl_matrix_get(W, i, j) +
            (gsl_matrix_get(X, k, i) *
            gsl_matrix_get(X, k, j) *
            p_k * (1-p_k));
          gsl_matrix_set(W, i, j, w_ij);
        }
        // Reflect (symmetry)
        if (i != j) {
          gsl_matrix_set(W, j, i, gsl_matrix_get(W, i, j));
        }
      }
    }
    gsl_linalg_LU_decomp(W, W_permut, &signum);
    gsl_linalg_LU_invert(W, W_permut, Wo);
    // b = b + Wo %*% X*(y-p)
    gsl_vector_set_zero(f);
    for(i = 0; i < X_cols; i++) {
      for(j = 0; j < X_rows; j++) {
        gsl_vector_set(f, i, gsl_vector_get(f, i) + gsl_matrix_get(X, j, i) * (gsl_vector_get(_y_dbl, j) - gsl_vector_get(_p, j)));
      }
    }
    gsl_blas_dgemv(CblasNoTrans, 1.0, Wo, f, 1.0, b);
    // Stopping condition
    max_rel_change = 0;
    for(i = 0; i < X_cols; i++) {
      bl_i = gsl_vector_get(bl, i);
      rel_change = fabs(gsl_vector_get(b, i) - bl_i) / (fabs(bl_i) + 0.01*_tol_irls);
      if (rel_change > max_rel_change) {
        max_rel_change = rel_change;
      }
    }
    if (max_rel_change < _tol_irls) {
      break;
    }
    gsl_vector_memcpy(bl, b);
  }

  // Calculate deviance
  double dev = 0;
  for(i = 0; i < X_rows; i++) {
    y_i = gsl_vector_get(_y_dbl, i);
    p_i = gsl_vector_get(_p, i);
    dev += (y_i * log(p_i)) + ((1-y_i)*log(1-p_i));
  }

  return dev;
}
