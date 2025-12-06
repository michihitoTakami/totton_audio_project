/**
 * OPRA Search Component - Shared Alpine.js functions
 *
 * Usage: Include these functions in your Alpine.js data object
 *
 * Required data properties:
 * - opra: { searchQuery, results, selected, eqProfiles, selectedEqId, useModernTarget, searching }
 */

/**
 * Search OPRA database for headphones
 */
async function searchOPRA() {
    if (!this.opra.searchQuery || this.opra.searchQuery.length < 2) {
        this.opra.results = [];
        return;
    }

    this.opra.searching = true;
    try {
        const response = await fetch(`/opra/search?q=${encodeURIComponent(this.opra.searchQuery)}`);
        if (!response.ok) throw new Error('Search failed');
        const data = await response.json();
        this.opra.results = data.results || [];
    } catch (error) {
        console.error('Failed to search OPRA:', error);
        this.opra.results = [];
    } finally {
        this.opra.searching = false;
    }
}

/**
 * Select a headphone from search results
 * @param {Object} result - Search result object with {id, name, vendor, eq_profiles}
 */
function selectHeadphone(result) {
    this.opra.selected = result;
    this.opra.results = [];
    this.opra.searchQuery = result.name;
    // eq_profiles is an array of {id, name, author}
    this.opra.eqProfiles = result.eq_profiles || [];
    this.opra.selectedEqId = this.opra.eqProfiles[0]?.id || '';
}

/**
 * Apply selected OPRA EQ profile
 * @param {Function} onSuccess - Callback function on success (optional)
 * @param {Function} onError - Callback function on error (optional)
 */
async function applyOPRA(onSuccess, onError) {
    if (!this.opra.selected || this.actionInProgress) return;

    this.actionInProgress = true;
    try {
        const eqId = this.opra.selectedEqId || this.opra.eqProfiles[0]?.id;
        if (!eqId) {
            throw new Error('No EQ profile selected');
        }

        const endpoint = `/opra/apply/${encodeURIComponent(eqId)}?apply_correction=${this.opra.useModernTarget}`;
        const response = await fetch(endpoint, {
            method: 'POST',
        });

        if (!response.ok) throw new Error('Failed to apply EQ');

        const data = await response.json();

        if (onSuccess) {
            onSuccess.call(this, data);
        } else {
            // Default success handler
            if (this.showToast) {
                this.showToast(data.message || 'EQを適用しました', 'success');
            }
        }
    } catch (error) {
        console.error('Failed to apply EQ:', error);

        if (onError) {
            onError.call(this, error);
        } else {
            // Default error handler
            if (this.showToast) {
                this.showToast('EQの適用に失敗しました', 'error');
            }
        }
    } finally {
        this.actionInProgress = false;
    }
}

// Export for use in Alpine.js components
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { searchOPRA, selectHeadphone, applyOPRA };
}
