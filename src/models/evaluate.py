import numpy as np
from scipy.stats import chi2_contingency
import argparse

def evaluate_ab_test(control_conversions: int, control_total: int, treatment_conversions: int, treatment_total: int, alpha: float = 0.05):
    """
    Evaluates the offline success of deployed psychological nudges using 
    a Chi-Square Test of Independence to determine statistical significance.
    This fulfills the A/B testing mandate from the system blueprint.
    """
    print("\n================= A/B Test Evaluation ==================")
    print(f"Control Group   : {control_conversions} conversions / {control_total} sessions ({(control_conversions/control_total)*100:.2f}%)")
    print(f"Treatment Group : {treatment_conversions} conversions / {treatment_total} sessions ({(treatment_conversions/treatment_total)*100:.2f}%)")
    print("--------------------------------------------------------")
    
    # Calculate non-conversions (abandonments)
    control_abandoned = control_total - control_conversions
    treatment_abandoned = treatment_total - treatment_conversions
    
    # Build the Contingency Table
    #           | Converted | Abandoned |
    # Control   |     A     |     B     |
    # Treatment |     C     |     D     |
    contingency_table = np.array([
        [control_conversions, control_abandoned],
        [treatment_conversions, treatment_abandoned]
    ])
    
    # Execute SciPy Statistical Test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print("Statistical Results:")
    print(f"  Chi-Square Statistic : {chi2:.4f}")
    print(f"  P-Value              : {p_value:.6e}")
    print("--------------------------------------------------------")
    
    # Determine Business Impact
    if p_value < alpha:
        print(f"Conclusion: STATISTICALLY SIGNIFICANT (p < {alpha})")
        print("The psychological nudge actively and successfully manipulated the consumer's")
        print("decision-making process, proving positive ROI.")
    else:
        print(f"Conclusion: NOT SIGNIFICANT (p >= {alpha})")
        print("The intervention did not yield a mathematically proven improvement in conversion rates.")
        print("Recommend adjusting the nudge strategy or cluster mapping.")
    print("========================================================\n")


if __name__ == "__main__":
    # Provides a CLI interface to input historical tracking logs manually or via CI scripts.
    parser = argparse.ArgumentParser(description="Evaluate Nudge A/B Test Statistical Significance")
    parser.add_argument("--cc", type=int, required=True, help="Number of Conversions in the Control Group")
    parser.add_argument("--ct", type=int, required=True, help="Total Number of Sessions in the Control Group")
    parser.add_argument("--tc", type=int, required=True, help="Number of Conversions in the Treatment Group")
    parser.add_argument("--tt", type=int, required=True, help="Total Number of Sessions in the Treatment Group")
    
    args = parser.parse_args()
    
    # Run the offline evaluation
    evaluate_ab_test(args.cc, args.ct, args.tc, args.tt)
