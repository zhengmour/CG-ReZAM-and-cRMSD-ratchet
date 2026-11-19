#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <thread>
#include <mutex>
#include <mpi.h>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <limits>
#include <stdexcept>
namespace py = pybind11;

#define XI 2 
#define YI 3 
#define ZI 4 


#define Stype 1
// All std::vector<double> store information as: id, type, x, y, z

// Constant used in Merge sort


// Sigmoid function in [0, 1] range
inline double sigmoid(double inx) {
    return 1.0 / (1.0 + std::exp(-inx));
}

std::vector<double> _get_center(const std::vector<std::vector<double>> &STRUCTURE) {
    std::vector<double> center = {0.0, 0.0, 0.0};
    const int atom_number = STRUCTURE.size();
	for (int i = 0; i < atom_number; ++i) {
		center[0] += STRUCTURE[i][XI];
		center[1] += STRUCTURE[i][YI];
		center[2] += STRUCTURE[i][ZI];
	}
	center[0] /= atom_number;
	center[1] /= atom_number;
	center[2] /= atom_number;
    return center;
}

// Spatial translation + rotation operation
std::vector<std::vector<double>> _move(
	const std::vector<std::vector<double>> &STRUCTURE, 
	double *DISPLACEMENT) {

	auto STR = STRUCTURE;
	const int atom_number = STR.size();

	// Compute geometric center
    std::vector<double> center = _get_center(STRUCTURE);

	// Vector -> matrix
	Eigen::MatrixXd T_P(4, atom_number);
    for (int i = 0; i < atom_number; ++i) {
        T_P(0, i) = STR[i][XI];
        T_P(1, i) = STR[i][YI];
        T_P(2, i) = STR[i][ZI];
        T_P(3, i) = 1.0;
    }

	// Translation matrices
    Eigen::Matrix4d T1, T2, T3;
    T1 << 1, 0, 0, -center[0],
          0, 1, 0, -center[1],
          0, 0, 1, -center[2],
          0, 0, 0, 1;

    T2 << 1, 0, 0, center[0],
          0, 1, 0, center[1],
          0, 0, 1, center[2],
          0, 0, 0, 1;

    T3 << 1, 0, 0, DISPLACEMENT[0],
          0, 1, 0, DISPLACEMENT[1],
          0, 0, 1, DISPLACEMENT[2],
          0, 0, 0, 1;

	// Rotation matrices
    Eigen::Matrix4d Rx, Ry, Rz;
    const double cos_x = std::cos(DISPLACEMENT[3]);
    const double sin_x = std::sin(DISPLACEMENT[3]);
    const double cos_y = std::cos(DISPLACEMENT[4]);
    const double sin_y = std::sin(DISPLACEMENT[4]);
    const double cos_z = std::cos(DISPLACEMENT[5]);
    const double sin_z = std::sin(DISPLACEMENT[5]);
    // Rotate around x-axis
    Rx << 1, 0, 0, 0,
          0, cos_x, -sin_x, 0,
          0, sin_x, cos_x, 0,
          0, 0, 0, 1;
    
    // Rotate around y-axis
    Ry << cos_y, 0, sin_y, 0,
          0, 1, 0, 0,
          -sin_y, 0, cos_y, 0,
          0, 0, 0, 1;
    
    // Rotate around z-axis
    Rz << cos_z, -sin_z, 0, 0,
          sin_z, cos_z, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1;

	auto R_P = T3 * T2 * Rz * Ry * Rx * T1 * T_P;

	for (int i = 0; i < atom_number; i++) {
		STR[i][XI] = R_P(0, i);
		STR[i][YI] = R_P(1, i);
		STR[i][ZI] = R_P(2, i);
	}

	return STR;
}

// Compare distances of two vectors from the geometric center
inline bool distance_pbc(const std::vector<double> &a, 
                const std::vector<double> &b, 
                const std::vector<double> &c) {

    const double dx1 = a[XI] - c[0];
    const double dy1 = a[YI] - c[1];
    const double dz1 = a[ZI] - c[2];
    
    const double dx2 = b[XI] - c[0];
    const double dy2 = b[YI] - c[1];
    const double dz2 = b[ZI] - c[2];
    
    const double r1_sq = dx1*dx1 + dy1*dy1 + dz1*dz1;
    const double r2_sq = dx2*dx2 + dy2*dy2 + dz2*dz2;
    
    return r1_sq < r2_sq;
}

// Stable sorting avoids the old four-column sentinel being read as five columns.
std::vector<std::vector<double>> _Sort2center(const std::vector<std::vector<double>> &base) {
    if (base.empty()) throw std::invalid_argument("Empty reference structure");
    for (const auto &row : base)
        if (row.size() != 5) throw std::invalid_argument("Expected [id,type,x,y,z]");
    auto sorted = base;
    const auto center = _get_center(base);
    std::stable_sort(sorted.begin(), sorted.end(), [&](const auto &a, const auto &b) {
        return distance_pbc(a, b, center);
    });
    return sorted;
}

double _sigmoidal_logistic(const double &r, const double &critical_radius, const double &alpha) {
    return 1.0 / (1.0 + std::exp(alpha*(r-critical_radius)));
}

// Compute per-frame RMSD, gradients and moved atom information
std::pair<double, std::vector<std::vector<double>>> _min_R2(
	const std::vector<std::vector<double>> &BASE_STRUCTURE, 
	const std::vector<std::vector<double>> &SOLVE_ATOM,
    double critical_radius, 
    double alpha) {

    if (BASE_STRUCTURE.empty() || SOLVE_ATOM.empty())
        throw std::invalid_argument("PIRMSD requires nonempty structures");
    std::unordered_map<int, std::vector<int>> by_type;
    for (int i=0; i<static_cast<int>(SOLVE_ATOM.size()); ++i)
        by_type[static_cast<int>(SOLVE_ATOM[i][1])].push_back(i);
    const auto center = _get_center(BASE_STRUCTURE);
    std::vector<unsigned char> used(SOLVE_ATOM.size(), 0);
    std::vector<std::vector<double>> moves;
    moves.reserve(std::min(BASE_STRUCTURE.size(), SOLVE_ATOM.size()));
    double sum=0., weights=0.;
    for (const auto &ref : BASE_STRUCTURE) {
        int best=-1;
        double d2=std::numeric_limits<double>::infinity();
        for (int j : by_type[static_cast<int>(ref[1])]) {
            if (used[j]) continue;
            const auto &atom=SOLVE_ATOM[j];
            double dx=atom[2]-ref[2], dy=atom[3]-ref[3], dz=atom[4]-ref[4];
            double candidate=dx*dx+dy*dy+dz*dz;
            if (candidate<d2) { d2=candidate; best=j; }
        }
        if (best<0) continue;
        used[best]=1;
        const auto &atom=SOLVE_ATOM[best];
        double r2=0.;
        for (int k=0;k<3;++k) r2+=std::pow(atom[k+2]-center[k],2);
        const double r=std::sqrt(r2);
        // Calculate expensive weights/derivatives once, after selecting a match.
        double w=critical_radius<1.e6 ? sigmoid(-alpha*(r-critical_radius)) : 1.;
        std::vector<double> row(9,0.);
        row[0]=atom[0]; row[1]=atom[1]; row[8]=w;
        for (int k=0;k<3;++k) {
            double dw=-alpha*w*(1.-w)*(atom[k+2]-center[k])/std::max(r,1.e-8);
            row[k+5]=dw;
            row[k+2]=2.*w*(atom[k+2]-ref[k+2])+d2*dw;
        }
        sum+=w*d2; weights+=w; moves.push_back(std::move(row));
    }
    if (moves.empty() || weights<=1.e-30)
        throw std::invalid_argument("No matching atoms with nonzero PIRMSD weight");
    return std::make_pair(std::sqrt(sum/weights), moves);
}

// Fixed reference slots: every vacancy has unit cost. Matched residuals use
// a C2 bounded loss.
std::pair<double, std::vector<std::vector<double>>> _boundary_R2(
    const std::vector<std::vector<double>> &base,
    const std::vector<std::vector<double>> &atoms, double cutoff) {
    if (base.empty() || !std::isfinite(cutoff) || cutoff<=0.)
        throw std::invalid_argument("Boundary PIRMSD needs reference slots and a positive cutoff");
    std::unordered_map<int, std::vector<int>> by_type;
    for (int i=0;i<static_cast<int>(atoms.size());++i)
        by_type[static_cast<int>(atoms[i][1])].push_back(i);
    std::vector<unsigned char> used(atoms.size(),0);
    std::vector<std::vector<double>> moves;
    // Accumulate loss directly so near-perfect matches do not lose precision.
    double sum=0.;
    for (const auto &ref : base) {
        int best=-1;
        double d2=std::numeric_limits<double>::infinity();
        for (int j : by_type[static_cast<int>(ref[1])]) {
            if (used[j]) continue;
            double candidate=0.;
            for (int k=2;k<5;++k) candidate+=std::pow(atoms[j][k]-ref[k],2);
            if (candidate<d2) { best=j; d2=candidate; }
        }
        if (best<0 || d2>=cutoff*cutoff) { sum+=1.; continue; }
        used[best]=1;
        double u=d2/(cutoff*cutoff), z=std::max(0.,1.-u);
        double loss=u<1. ? u*(3.-3.*u+u*u) : 1.;
        sum+=loss;
        std::vector<double> row(10,0.);
        row[0]=atoms[best][0]; row[1]=atoms[best][1]; row[8]=1.; row[9]=ref[0];
        for (int k=0;k<3;++k)
            row[k+2]=6.*z*z*(atoms[best][k+2]-ref[k+2])/(cutoff*cutoff);
        moves.push_back(std::move(row));
    }
    return std::make_pair(std::sqrt(sum/base.size()),moves);
}

// MPI version of BFGS optimization function
std::tuple<double, std::vector<std::vector<double>>, std::vector<double>> _BFGS_CENTER(
	const std::vector<std::vector<double>> &BASE_STRUCTURE, 
	const std::vector<std::vector<double>> &SOLVE_ATOM, 
	int OUTER, 
	int INTER, 
	double Epsilon, 
	std::vector<double> &MOVE, 
	bool fixed_move,
    double critical_radius,
    double alpha,
    int f_comm, double boundary_cutoff=0.) {

	if (OUTER < 1 || INTER < 1 || MOVE.size()!=6 || Epsilon<=0)
        throw std::invalid_argument("Invalid PIRMSD optimizer settings");

	auto bs = BASE_STRUCTURE;
	auto sa = SOLVE_ATOM;
	auto mv = MOVE;
    auto evaluate = [&](const std::vector<std::vector<double>> &reference) {
        return boundary_cutoff>0. ? _boundary_R2(reference, sa, boundary_cutoff)
                                  : _min_R2(reference, sa, critical_radius, alpha);
    };

	std::array<double, 6> move0 = {mv[0], mv[1], mv[2], mv[3], mv[4], mv[5]};

    // Get MPI communicator
    MPI_Comm comm = MPI_Comm_f2c(f_comm);

    // MPI initialization
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Whether to perform optimization iterations
	if (fixed_move) {
        auto temp_bs = _move(bs, move0.data());
        auto D_P = evaluate(temp_bs);
        return std::make_tuple(D_P.first, D_P.second, mv);
	}

    // Compute number of OUTER iterations per process
    const int local_outer = (OUTER + size - 1) / size;  // Ceil division
    const int start_idx = rank * local_outer;
    const int end_idx = std::min(start_idx + local_outer, OUTER);
    const int actual_local_outer = std::max(0, end_idx - start_idx);

    // Retain one candidate per rank, including the incoming warm start.
    auto initial = evaluate(_move(bs, move0.data()));
    double local_best = initial.first;
    std::array<double,6> local_move = move0;

    if (actual_local_outer > 0) {
        Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();

        // Random number generator
        std::mt19937 gen(1729 + rank);  // Different seed for each process
        std::uniform_real_distribution<double> distrib(0.0, 1.0);

        // Local outer-loop iterations
        for (int local_outer_idx = 0; local_outer_idx < actual_local_outer; ++local_outer_idx) {
			double RMSD = 10000.0;
            double fk = 1000000.0;
            double fk_1;

            std::array<double, 6> xk = {0.0, 0.0, 0.0, 
                                        (distrib(gen) - 0.5) * 3.14159, 
                                        (distrib(gen) - 0.5) * 3.14159, 
                                        (distrib(gen) - 0.5) * 3.14159};
            std::array<double, 6> xk_1;
            Eigen::Matrix<double, 6, 1> dk, sk, yk, gk_1, gk;
            Eigen::Matrix<double, 6, 6> Dk = I, Dk_1;

            auto objective = [&](std::array<double,6> x) {
                return evaluate(_move(bs, x.data())).first;
            };
            auto gradient = [&](const std::array<double,6> &x) {
                Eigen::Matrix<double,6,1> g = Eigen::Matrix<double,6,1>::Zero();
                for (int j=3;j<6;++j) {
                    auto plus=x, minus=x;
                    plus[j]+=1.e-5; minus[j]-=1.e-5;
                    g[j]=(objective(plus)-objective(minus))/2.e-5;
                }
                return g;
            };
            // Rotation only: translation is handled by Python centering.
            if (start_idx+local_outer_idx==0) xk=move0;
            fk=objective(xk);
            if (fk<local_best) { local_best=fk; local_move=xk; }
            gk=gradient(xk);

            // Inner iterations
            for (int inter = 0; inter < INTER; ++inter) {
                dk = -Dk * gk;

                if (dk.dot(gk)>=0.) { Dk=I; dk=-gk; }
                double step=std::min(1., 0.25/std::max(dk.norm(),1.e-30));
                bool improved=false;
                for (int trial=0;trial<12;++trial) {
                    for (int j=0;j<6;++j) xk_1[j]=xk[j]+step*dk[j];
                    fk_1=objective(xk_1);
                    if (fk_1<=fk+1.e-4*step*gk.dot(dk)) { improved=true; break; }
                    step*=0.5;
                }
                if (!improved) break;
                for (int j=0;j<6;++j) sk[j]=xk_1[j]-xk[j];
                RMSD=fk_1;
                gk_1=gradient(xk_1);
                yk=gk_1-gk;

                // Convergence check
                double Ngk_1 = gk_1.norm();
                if (RMSD < local_best) { local_best = RMSD; local_move = xk_1; }

                if (Ngk_1 <= Epsilon) {
                    break;
                }

                // Update Hessian approximation
                double L = yk.transpose() * sk;
                if (L <= 1.e-12) {
                    Dk_1 = I;
                } else {
                    const auto I_minus_sk_yk = I - (sk * yk.transpose()) / L;
                    const auto I_minus_yk_sk = I - (yk * sk.transpose()) / L;
                    Dk_1 = I_minus_sk_yk * Dk * I_minus_yk_sk + (sk * sk.transpose()) / L;
                }

                // Update parameters
                xk = xk_1;
                fk = fk_1;
				for (int i = 0; i < 6; i++)
					gk(i, 0) = gk_1(i, 0);
                Dk = Dk_1;
            }
        }
    }

    struct { double value; int rank; } local{local_best, rank}, global;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
    MPI_Bcast(local_move.data(), 6, MPI_DOUBLE, global.rank, comm);
    auto result = evaluate(_move(bs, local_move.data()));
    return std::make_tuple(result.first, result.second,
                           std::vector<double>(local_move.begin(), local_move.end()));
}

std::tuple<double, std::vector<std::vector<double>>, std::vector<double>> _BFGS_NORMAL(
	const std::vector<std::vector<double>> &BASE_STRUCTURE, 
	const std::vector<std::vector<double>> &SOLVE_ATOM, 
	int OUTER, 
	int INTER, 
	double Epsilon, 
	std::vector<double> &MOVE, 
	bool fixed_move,
    int f_comm) {
    
    return _BFGS_CENTER(BASE_STRUCTURE, SOLVE_ATOM, OUTER, INTER, Epsilon, MOVE, fixed_move, 1E7, 10, f_comm);
}

PYBIND11_MODULE(C_PIRMSD, m)
{
	m.attr("API_VERSION") = 3;
	m.doc() = "C_PIRMSD"; // optional module docstring
	m.def("_min_R2", &_min_R2, "A function that computing min_RMSD");
	m.def("_Sort2center", &_Sort2center, "A function that computing min_RMSD");
	m.def("_BFGS_NORMAL", &_BFGS_NORMAL, "A function that computing min_RMSD");
	m.def("_BFGS_CENTER", [](const std::vector<std::vector<double>> &base,
        const std::vector<std::vector<double>> &atoms, int outer, int inner,
        double epsilon, std::vector<double> move, bool fixed, double radius,
        double alpha, int comm) {
        return _BFGS_CENTER(base,atoms,outer,inner,epsilon,move,fixed,radius,alpha,comm);
    });
    m.def("_boundary_R2", &_boundary_R2);
    m.def("_BFGS_BOUNDARY", [](const std::vector<std::vector<double>> &base,
        const std::vector<std::vector<double>> &atoms, int outer, int inner,
        double epsilon, std::vector<double> move, bool fixed, double cutoff, int comm) {
        if (!std::isfinite(cutoff) || cutoff<=0.) throw std::invalid_argument("Invalid cutoff");
        return _BFGS_CENTER(base,atoms,outer,inner,epsilon,move,fixed,1.e7,0.,comm,cutoff);
    });
}
