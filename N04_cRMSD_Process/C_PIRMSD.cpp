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
namespace py = pybind11;

#define XI 2 
#define YI 3 
#define ZI 4 

std::mutex mtx; // Protect counter

#define Stype 1
// All std::vector<double> store information as: id, type, x, y, z

// Constant used in Merge sort
const std::vector<double> Max = {-1.0, 100000.0, 100000.0, 100000.0};

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

// Merge sort merge step
void Merge(std::vector<std::vector<double>> &Array, 
           const std::vector<double> &c, int front, int mid, int end) {
    
    std::vector<std::vector<double>> LeftSub(
        Array.begin() + front, Array.begin() + mid + 1);
    std::vector<std::vector<double>> RightSub(
        Array.begin() + mid + 1, Array.begin() + end + 1);

    LeftSub.push_back(Max);
    RightSub.push_back(Max);

    int idxLeft = 0, idxRight = 0;

    for (int i = front; i <= end; ++i) {
        if (distance_pbc(LeftSub[idxLeft], RightSub[idxRight], c)) {
            Array[i] = std::move(LeftSub[idxLeft]);
            ++idxLeft;
        } else {
            Array[i] = std::move(RightSub[idxRight]);
            ++idxRight;
        }
    }
}
void MergeSort(std::vector<std::vector<double>> &array, 
               const std::vector<double> &c, int front, int end) {
    if (front < end) {
        const int mid = front + (end - front) / 2;  // Prevent overflow
        MergeSort(array, c, front, mid);
        MergeSort(array, c, mid + 1, end);
        Merge(array, c, front, mid, end);
    }
}

// Sort reference structure by distance to geometric center
std::vector<std::vector<double>> _Sort2center(const std::vector<std::vector<double>> &BASE_STRUCTURE) {
	auto bs = BASE_STRUCTURE;

    // Compute geometric center
    std::vector<double> center = _get_center(BASE_STRUCTURE);

    // Sort
    MergeSort(bs, center, 0, bs.size() - 1);
    return bs;
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

    // Group atoms by type
    std::unordered_map<int, std::vector<int>> atoms_by_type;
    for (int i = 0; i < SOLVE_ATOM.size(); ++i) {
        int atom_type = static_cast<int>(SOLVE_ATOM[i][1]);
        atoms_by_type[atom_type].push_back(i);
    }

	// Output parameters
	double RMSD = 0;
	std::vector<std::vector<double>> MOVE_ATOM;
	std::unordered_set<int> selected_atom;  // Use unordered_set to speed up lookup

    const auto& base = BASE_STRUCTURE;
    const auto& atoms = SOLVE_ATOM;
    const std::vector<double> center = _get_center(BASE_STRUCTURE);

    double weight = 1.0;
    double delta_x, delta_y, delta_z, d2, delta_x_c, delta_y_c, delta_z_c, r, safe_r, exp_term;  
    double dw_dr, dr_dx, dr_dy, dr_dz, dw_dx, dw_dy, dw_dz, grad_dx, grad_dy, grad_dz;
    double sum_weight = 0.0;

	MOVE_ATOM.reserve(base.size());

	for (auto &refer : base) {
        if (selected_atom.size() >= atoms.size()) break;

		double dmin2 = INFINITY;
		std::vector<double> choose_atom; // id, type, dSx, dSy, dSz, dWx, dWy, dWz, weight

		const int refer_type = static_cast<int>(refer[1]);
        if (atoms_by_type.find(refer_type) != atoms_by_type.end()) {
            for (int atom_idx : atoms_by_type[refer_type]) {
                // Check whether we have already used all available atoms
                if (selected_atom.size() >= atoms.size()) break;

                const auto &atom = atoms[atom_idx];
                const int atom_type = static_cast<int>(atom[1]);
                if (selected_atom.count(static_cast<int>(atom[0]))) continue;  // Skip already selected atoms
                
                // // Fast distance estimate (Manhattan distance pre-filter)
                // dx_abs = std::abs(refer_x - atom[XI]);
                // dy_abs = std::abs(refer_y - atom[YI]);
                // dz_abs = std::abs(refer_z - atom[ZI]);
                //
                // if (dx_abs + dy_abs + dz_abs > max_search_distance * 1.73) continue; // Rough pre-filter

                delta_x = atom[XI] - refer[XI];
                delta_y = atom[YI] - refer[YI];
                delta_z = atom[ZI] - refer[ZI];
                d2 = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z;
                
                if (d2 < dmin2) {	
                    dmin2 = d2;     
                    choose_atom.clear();		
                    choose_atom.resize(9);	
                    weight = 1.0; dw_dx = 0.0; dw_dy = 0.0; dw_dz = 0.0;
                    
                    if (critical_radius < 1.0E6) {                        
                        delta_x_c = atom[XI] - center[0];		
                        delta_y_c = atom[YI] - center[1];		
                        delta_z_c = atom[ZI] - center[2];	
                        r = sqrt(delta_x_c*delta_x_c + delta_y_c*delta_y_c + delta_z_c*delta_z_c);
                        
                        safe_r = (r < 1e-8) ? 1e-8 : r;
                        exp_term = std::exp(alpha * (r - critical_radius));
                        weight = 1.0 / (1.0 + exp_term);
                        dw_dr = -alpha * weight * (1.0 - weight);

                        // 4. Compute derivative of distance
                        dr_dx = delta_x_c / safe_r;
                        dr_dy = delta_y_c / safe_r;
                        dr_dz = delta_z_c / safe_r;
                        
                        // 5. Compute derivative of weight with respect to coordinates
                        dw_dx = dw_dr * dr_dx;
                        dw_dy = dw_dr * dr_dy;
                        dw_dz = dw_dr * dr_dz;
                    } else {
                        weight = 1.0; dw_dx = 0.0; dw_dy = 0.0; dw_dz = 0.0;
                    }

                    // dS/dx_j = 2*w*dx + d^2 * (dw/dx_j)
                    grad_dx = 2.0 * weight * delta_x + d2 * dw_dx;
                    grad_dy = 2.0 * weight * delta_y + d2 * dw_dy;
                    grad_dz = 2.0 * weight * delta_z + d2 * dw_dz;

                    choose_atom[0] = atom[0];
                    choose_atom[1] = atom_type;
                    choose_atom[2] = grad_dx;
                    choose_atom[3] = grad_dy;
                    choose_atom[4] = grad_dz;
                    choose_atom[5] = dw_dx;
                    choose_atom[6] = dw_dy;
                    choose_atom[7] = dw_dz;
                    choose_atom[8] = weight;
            
                }    	
            }
            if (!choose_atom.empty()) { 
                selected_atom.insert(static_cast<int>(choose_atom[0]));
                sum_weight += choose_atom[8];
                RMSD += weight * dmin2;
                MOVE_ATOM.push_back(std::move(choose_atom));    
            }
        }
    }

    const double eps = 1e-30;
    RMSD = sqrt(RMSD / std::max(sum_weight, eps));
    return make_pair(RMSD, MOVE_ATOM);
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
    int f_comm) {

	// Generate different random seeds
	std::random_device rd;  // If available, get a true random seed from the random device
	std::mt19937 gen(rd()); // Standard Mersenne Twister engine seeded with rd()
	std::uniform_real_distribution<double> distrib(0, 1);

	auto bs = BASE_STRUCTURE;
	auto sa = SOLVE_ATOM;
	auto mv = MOVE;

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
        auto D_P = _min_R2(temp_bs, sa, critical_radius, alpha);
        return std::make_tuple(D_P.first, D_P.second, mv);
	}

    // Compute number of OUTER iterations per process
    const int local_outer = (OUTER + size - 1) / size;  // Ceil division
    const int start_idx = rank * local_outer;
    const int end_idx = std::min(start_idx + local_outer, OUTER);
    const int actual_local_outer = std::max(0, end_idx - start_idx);

    // Local result storage
    std::vector<double> local_RMSD_list;
    std::vector<std::vector<std::vector<double>>> local_MOVE_ATOM_list;
    std::vector<std::array<double, 6>> local_MOVE_list;

    if (actual_local_outer > 0) {
        local_RMSD_list.resize(actual_local_outer * INTER, 10000.0);
        local_MOVE_ATOM_list.resize(actual_local_outer * INTER);
        local_MOVE_list.resize(actual_local_outer * INTER);

        // Initialize xk, fk, bs
        auto temp_bs = _move(bs, move0.data());
        std::array<double, 6> x0 = {0, 0, 0, 0, 0, 0};
        double f0 = _min_R2(temp_bs, sa, critical_radius, alpha).first;

        Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();

        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd() + rank);  // Different seed for each process
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
            std::array<double, 6> lambda = {0.0, 0.0, 0.0, 
                                            distrib(gen) * 0.1, 
                                            distrib(gen) * 0.1, 
                                            distrib(gen) * 0.1};

            std::vector<std::vector<double>> MOVE_ATOM;

            Eigen::Matrix<double, 6, 1> dk, sk, yk, gk_1, gk;
            Eigen::Matrix<double, 6, 6> Dk = I, Dk_1;

            // Initialize gradient
            for (int i = 0; i < 6; ++i) {
                const double diff = xk[i] - x0[i];
                if (std::abs(diff) > 1e-10) {
                    const double grad = (fk - f0) / diff;
                    gk(i, 0) = std::clamp(grad, -10.0, 10.0);  // Clamp gradient range
                } else {
                    gk(i, 0) = Epsilon;
                }
            }

            // Inner iterations
            for (int inter = 0; inter < INTER; ++inter) {
                dk = -Dk * gk;

                // Use sigmoid function to limit the step size
                for (int i = 0; i < 6; ++i) {
                    dk(i, 0) = 2.0 * (sigmoid(dk(i, 0)) - 0.5);
                    sk(i, 0) = lambda[i] * dk(i, 0);
                    xk_1[i] = xk[i] + sk(i, 0);
                }

                auto temp_bs = _move(bs, xk_1.data());
                auto D_P = _min_R2(temp_bs, sa, critical_radius, alpha);
                RMSD = D_P.first;
                fk_1 = D_P.first;
                MOVE_ATOM = D_P.second;

                // Compute new gradient
                for (int i = 0; i < 6; ++i) {
                    const double diff = xk_1[i] - xk[i];
                    if (std::abs(diff) > 1e-10) {
                        const double grad = (fk_1 - fk) / diff;
                        gk_1(i, 0) = std::clamp(grad, -10.0, 10.0);
                    } else {
                        gk_1(i, 0) = Epsilon;
                    }
                    yk(i, 0) = gk_1(i, 0) - gk(i, 0);
                }

                // Convergence check
                double Ngk_1 = gk_1.norm();
                const int result_idx = local_outer_idx * INTER + inter;
                local_RMSD_list[result_idx] = RMSD;
                local_MOVE_ATOM_list[result_idx] = MOVE_ATOM;
                local_MOVE_list[result_idx] = xk_1;

                if (Ngk_1 <= Epsilon) {
                    break;
                }

                // Update Hessian approximation
                double L = yk.transpose() * sk;
                if (std::abs(L) < Epsilon) {
                    // Reset Hessian approximation
                    for (int i = 0; i < 6; i++) 
						for (int j = 0; j < 6; j++) 
								Dk_1(i, j) = (sigmoid(Dk(i, j)) - 0.5) / Epsilon;
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

	// MPI collect results
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    int local_count = local_RMSD_list.size();
    MPI_Allgather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    
    int total_count = 0;
    for (int i = 0; i < size; ++i) {
        displs[i] = total_count;
        total_count += recvcounts[i];
    }

	std::vector<double> global_RMSD_list(total_count);
    MPI_Allgatherv(local_RMSD_list.data(), local_count, MPI_DOUBLE,
                   global_RMSD_list.data(), recvcounts.data(), displs.data(), 
                   MPI_DOUBLE, comm);

    // Find the best (minimum) solution
    auto smallest = std::min_element(global_RMSD_list.begin(), global_RMSD_list.end());
    int global_minID = std::distance(global_RMSD_list.begin(), smallest);
    
    // Determine which rank owns the best solution
    int owner_rank = 0;
    int local_minID = 0;
    int cumulative = 0;
    for (int i = 0; i < size; ++i) {
        if (global_minID < cumulative + recvcounts[i]) {
            owner_rank = i;
            local_minID = global_minID - cumulative;
            break;
        }
        cumulative += recvcounts[i];
    }

    double best_RMSD = global_RMSD_list[global_minID];
    std::vector<std::vector<double>> best_MOVE_ATOM;
    std::array<double, 6> best_move;

	// Broadcast best result from the owner rank
    if (rank == owner_rank && local_minID < local_MOVE_ATOM_list.size()) {
        best_MOVE_ATOM = local_MOVE_ATOM_list[local_minID];
        best_move = local_MOVE_list[local_minID];
    }

    // Broadcast size information of best_MOVE_ATOM
    int move_atom_size = (rank == owner_rank) ? best_MOVE_ATOM.size() : 0;
    MPI_Bcast(&move_atom_size, 1, MPI_INT, owner_rank, comm);
    
    if (rank != owner_rank) {
        best_MOVE_ATOM.resize(move_atom_size);
    }

    MPI_Bcast(best_move.data(), 6, MPI_DOUBLE, owner_rank, comm);

    // For non-owner ranks, recompute best_MOVE_ATOM locally
    if (rank != owner_rank) {
        // Recompute best solution
        auto temp_bs = _move(BASE_STRUCTURE, best_move.data());
        auto temp_result = _min_R2(temp_bs, sa, critical_radius, alpha);
        best_MOVE_ATOM = temp_result.second;

        // Check whether temp_result.first and best_RMSD are consistent
        double local_rmsd = temp_result.first;
        double diff = std::abs(local_rmsd - best_RMSD);

        if (diff > 1e-4) {  // Or set threshold based on precision requirement
            std::cerr << "[Rank " << rank << "] Warning: Recomputed RMSD (" << local_rmsd 
                    << ") differs from best_RMSD (" << best_RMSD 
                    << ") by " << diff << std::endl;
        }
    }

    // Update output parameter MOVE
    for (int i = 0; i < 6; ++i) {
        mv[i] = best_move[i];
    }

    return std::make_tuple(best_RMSD, best_MOVE_ATOM, mv);
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
	m.doc() = "C_PIRMSD"; // optional module docstring
	m.def("_min_R2", &_min_R2, "A function that computing min_RMSD");
	m.def("_Sort2center", &_Sort2center, "A function that computing min_RMSD");
	m.def("_BFGS_NORMAL", &_BFGS_NORMAL, "A function that computing min_RMSD");
	m.def("_BFGS_CENTER", &_BFGS_CENTER, "A function that computing min_RMSD");
}
