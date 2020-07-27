#include "../include.h"

/*
 * Minimize a continuous differentialble multivariate function. Starting point <br/>
 * is given by "X" (D by 1), and the function named in the string "f", must<br/>
 * return a function value and a vector of partial derivatives. The Polack-<br/>
 * Ribiere flavour of conjugate gradients is used to compute search directions,<br/>
 * and a line search using quadratic and cubic polynomial approximations and the<br/>
 * Wolfe-Powell stopping criteria is used together with the slope ratio method<br/>
 * for guessing initial step sizes. Additionally a bunch of checks are made to<br/>
 * make sure that exploration is taking place and that extrapolation will not<br/>
 * be unboundedly large. The "length" gives the length of the run: if it is<br/>
 * positive, it gives the maximum number of line searches, if negative its<br/>
 * absolute gives the maximum allowed number of function evaluations. You can<br/>
 * (optionally) give "length" a second component, which will indicate the<br/>
 * reduction in function value to be expected in the first line-search (defaults<br/>
 * to 1.0). The function returns when either its length is up, or if no further<br/>
 * progress can be made (ie, we are at a minimum, or so close that due to<br/>
 * numerical problems, we cannot get any closer). If the function terminates<br/>
 * within a few iterations, it could be an indication that the function value<br/>
 * and derivatives are not consistent (ie, there may be a bug in the<br/>
 * implementation of your "f" function). The function returns the found<br/>
 * solution "X", a vector of function values "fX" indicating the progress made<br/>
 * and "i" the number of iterations (line searches or function evaluations,<br/>
 * depending on the sign of "length") used.<br/>
 * <br/>
 * Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)<br/>
 * <br/>
 * See also: checkgrad <br/>
 * <br/>
 * Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13<br/>
 * <br/>
 * <br/>
 * (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen <br/>
 * Permission is granted for anyone to copy, use, or modify these<br/>
 * programs and accompanying documents for purposes of research or<br/>
 * education, provided this copyright notice is retained, and note is<br/>
 * made of any changes that have been made.<br/>
 * <br/>
 * These programs and documents are distributed without any warranty,<br/>
 * express or implied. As the programs were written for research<br/>
 * purposes only, they have not been tested to the degree that would be<br/>
 * advisable in any important application. All use of these programs is<br/>
 * entirely at the user's own risk.<br/>
 * <br/>
 * [ml-class] Changes Made:<br/>
 * 1) Function name and argument specifications<br/>
 * 2) Output display<br/>
 * <br/>
 * [tjungblut] Changes Made: <br/>
 * 1) translated from octave to java<br/>
 * 2) added an interface to exchange minimizers more easily <br/>
 * 3) in preparation for the c++ translation, I removed unused fields<br/>
 * 4) c++ implementation <br/>
 * BTW "fmincg" stands for Function minimize nonlinear conjugate gradient
 * [hud]
 * 1) move vector class to Eigen
 * 2) adjust types
 */

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            Math::T_Vector Fmincg::minimize(
                    Trainer::StepFunction stepFunction,
                    Math::T_Vector theta,
                    T_Size length,
                    bool verbose
            ) {
                // input will be the pointer to our current active parameter set
                Math::T_Vector input(std::move(theta));
                Math::T_Vector X0(input);
                // search directions
                Math::T_Vector s(input.size());
                // gradients
                Math::T_Vector df0(input.size());
                Math::T_Vector df1(input.size());
                Math::T_Vector df2(input.size());

                // define some integers for bookkeeping and then start
                int M = 0;
                int i = 0; // zero the run length counter
                int red = 1; // starting point
                int ls_failed = 0; // no previous line search has failed
                Impulse::NeuralNetwork::Trainer::CostGradientResult evaluateCost = stepFunction(input);
                double f1 = evaluateCost.getCost();
                df1 = evaluateCost.getGradient();
                i = i + (length < 0 ? 1 : 0);
                // search direction is steepest
                s = (df1 * -1.0);

                double d1 = (s * -1.0).dot(s); // this is the slope
                double z1 = red / (1.0 - d1); // initial step is red/(|s|+1)

                while (i < length) { // while not finished
                    high_resolution_clock::time_point begin = high_resolution_clock::now();
                    i = i + (length > 0 ? 1 : 0); // count iterations?!
                    // make a copy of current values
                    X0 = input;
                    double f0 = f1;
                    df0 = df1;
                    // begin line search
                    // fill our new line searched parameters
                    input = input + (s * z1);
                    Impulse::NeuralNetwork::Trainer::CostGradientResult evaluateCost2 = stepFunction(input);
                    double f2 = evaluateCost2.getCost();
                    df2 = evaluateCost2.getGradient();
                    i = i + (length < 0 ? 1 : 0); // count epochs
                    double d2 = df2.dot(s);
                    // initialize point 3 equal to point 1
                    double f3 = f1;
                    double d3 = d1;
                    double z3 = -z1;
                    if (length > 0) {
                        M = _MAX;
                    } else {
                        M = std::min(_MAX, static_cast<int>(-length) - i);
                    }
                    // initialize quanteties
                    int success = 0;
                    double limit = -1;

                    while (true) {
                        while (((f2 > f1 + z1 * _RHO * d1) | (d2 > -_SIG * d1)) && (M > 0)) {
                            // tighten the bracket
                            limit = z1;
                            double z2 = 0.0;
                            double A = 0.0;
                            double B = 0.0;
                            if (f2 > f1) {
                                // quadratic fit
                                z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
                            } else {
                                // cubic fit
                                A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
                                B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                                // numerical errorForSample possible - ok!
                                z2 = (sqrt(B * B - A * d2 * z3 * z3) - B) / A;
                            }
                            if (std::isnan(z2) || !finite(z2)) {
                                // if we had a numerical problem then bisect
                                z2 = z3 / 2.0;
                            }
                            // don't accept too close to limits
                            z2 = std::max(std::min(z2, _INT * z3), (1 - _INT) * z3);
                            // update the step
                            z1 = z1 + z2;
                            input = input + (s * z2);
                            Impulse::NeuralNetwork::Trainer::CostGradientResult evaluateCost3 = stepFunction(input);
                            f2 = evaluateCost3.getCost();
                            df2 = evaluateCost3.getGradient();
                            M = M - 1;
                            i = i + (length < 0 ? 1 : 0); // count epochs
                            d2 = df2.dot(s);
                            // z3 is now relative to the location of z2
                            z3 = z3 - z2;
                        }

                        if (f2 > f1 + z1 * _RHO * d1 || d2 > -_SIG * d1) {
                            break; // this is a failure
                        } else if (d2 > _SIG * d1) {
                            success = 1;
                            break; // success
                        } else if (M == 0) {
                            break; // failure
                        }
                        // make cubic extrapolation
                        double A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
                        double B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                        double z2 = -d2 * z3 * z3 / (B + sqrt(B * B - A * d2 * z3 * z3));
                        // num prob or wrong sign?
                        if (std::isnan(z2) || !finite(z2) || z2 < 0)
                            // if we have no upper limit
                            if (limit < -0.5) {
                                // the extrapolate the maximum amount
                                z2 = z1 * (_EXT - 1);
                            } else {
                                // otherwise bisect
                                z2 = (limit - z1) / 2;
                            }
                        else if ((limit > -0.5) && (z2 + z1 > limit)) {
                            // extraplation beyond max?
                            z2 = (limit - z1) / 2; // bisect
                        } else if ((limit < -0.5) && (z2 + z1 > z1 * _EXT)) {
                            // extrapolationbeyond limit
                            z2 = z1 * (_EXT - 1.0); // set to extrapolation limit
                        } else if (z2 < -z3 * _INT) {
                            z2 = -z3 * _INT;
                        } else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - _INT))) {
                            // too close to the limit
                            z2 = (limit - z1) * (1.0 - _INT);
                        }
                        // set point 3 equal to point 2
                        f3 = f2;
                        d3 = d2;
                        z3 = -z2;
                        z1 = z1 + z2;
                        // update current estimates
                        input = input + (s * z2);
                        Impulse::NeuralNetwork::Trainer::CostGradientResult evaluateCost3 = stepFunction(input);
                        f2 = evaluateCost3.getCost();
                        df2 = evaluateCost3.getGradient();
                        M = M - 1;
                        i = i + (length < 0 ? 1 : 0); // count epochs?!
                        d2 = df2.dot(s);
                    } // end of line search

                    if (success == 1) { // if line search succeeded
                        f1 = f2;
                        if (verbose) {
                            high_resolution_clock::time_point end = high_resolution_clock::now();
                            auto duration = duration_cast<milliseconds>(end - begin).count();
                            std::cout << "Iteration: " << i
                                      << " | Error: " << f1
                                      << " | Time: " << duration
                                      << std::endl;
                        }
                        // Polack-Ribiere direction: s =
                        // (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;
                        double df2len = df2.dot(df2);
                        double df12len = df1.dot(df2);
                        double df1len = df1.dot(df1);
                        double numerator = (df2len - df12len) / df1len;
                        s = ((s * numerator) - df2);
                        std::swap(df1, df2); // swap derivatives
                        d2 = df1.dot(s);
                        // new slope must be negative
                        if (d2 > 0) {
                            // otherwise use steepest direction
                            s = (df1 * -1.0);
                            d2 = (s * -1.0).dot(s);
                        }
                        // realmin in octave = 2.2251e-308
                        // slope ratio but max _RATIO
                        double thres = d1 / (d2 - 2.2251e-308);
                        z1 = z1 * std::min(_RATIO, thres);
                        d1 = d2;
                        ls_failed = 0; // this line search did not fail
                    } else {
                        // restore data from the beginning of the iteration
                        input = X0;
                        f1 = f0;
                        df1 = df0; // restore point from before failed line search
                        // line search failed twice in a row?
                        if (ls_failed == 1 || i > length) {
                            break; // or we ran out of time, so we give up
                        }
                        // swap derivatives
                        std::swap(df1, df2);
                        // try steepest
                        s = (df1 * -1.0);
                        d1 = (s * -1.0).dot(s);
                        z1 = 1.0 / (1.0 - d1);
                        ls_failed = 1; // this line search failed
                    }
                } // while true loop
                if (verbose) {
                    // if verbose, we want to switch to a newline now
                    std::cout << std::endl;
                }

                return input;
            }
        }
    }
}